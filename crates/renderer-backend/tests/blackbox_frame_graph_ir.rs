// Blackbox contract tests for Frame Graph IR (T-FG-1.1).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-FG-1.1):
//   IrPass, IrResource, IrEdge Rust structs with full field coverage,
//   pass name, type, resource access sets, colour attachments,
//   depth stencil. Write blackbox tests against public API contract.
//
// Coverage:
//   1.  ResourceHandle construction, NONE sentinel, Display, ordering
//   2.  PassIndex construction, Display
//   3.  PassType variant discrimination and Display
//   4.  ResourceAccess variant discrimination and Display
//   5.  ResourceAccessEntry construction and Display
//   6.  ResourceAccessSet empty/is_empty/len/contains/Display
//   7.  AttachmentLoadOp/AttachmentStoreOp variants and Display
//   8.  ColorAttachment -- full field construction, default, Display
//   9.  DepthStencilAttachment -- full field construction, default, Display
//  10.  InstanceSource -- Direct/Indirect/Mesh variants, field access, Display
//  11.  DispatchSource -- Direct/Indirect variants, field access, Display
//  12.  ViewType variant discrimination and Display
//  13.  TextureDesc/Texture3DDesc/BufferDesc construction, Display
//  14.  ResourceDesc variant discrimination, field extraction, Display
//  15.  ResourceLifetime/ResourceState variant access and Display
//  16.  IrResource -- new(), all fields, Display, view_format_override
//  17.  IrPass -- graphics/compute/copy/ray_tracing constructors
//  18.  IrPass -- sync_access_set_from_attachments
//  19.  IrPass -- has_color_attachments / has_depth_stencil / has_dispatch
//  20.  IrPass -- tags field, Display
//  21.  EdgeType variant discrimination and Display
//  22.  IrEdge -- new(), all fields, Display
//  23.  Integration: pass-resource-edge round trip with all IR types

use renderer_backend::frame_graph::{
    AttachmentLoadOp, AttachmentStoreOp, BufferDesc, ColorAttachment,
    DepthStencilAttachment, DispatchSource, EdgeType, InstanceSource, IrEdge,
    IrPass, IrResource, PassIndex, PassType, ResourceAccess,
    ResourceAccessEntry, ResourceAccessSet, ResourceDesc, ResourceHandle,
    ResourceLifetime, ResourceState, Texture3DDesc, TextureDesc, ViewType,
};

// =============================================================================
// SECTION 1 -- ResourceHandle: construction, sentinel, Display, ordering
// =============================================================================

#[test]
fn resource_handle_constructs_with_raw_u32() {
    let h = ResourceHandle(7);
    assert_eq!(h.0, 7, "ResourceHandle wraps a u32");
}

#[test]
fn resource_handle_none_sentinel_is_u32_max() {
    assert_eq!(
        ResourceHandle::NONE,
        ResourceHandle(u32::MAX),
        "NONE must be u32::MAX"
    );
}

#[test]
fn resource_handle_copy_and_clone_preserve_value() {
    let a = ResourceHandle(42);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b, "Copy preserves value");
    assert_eq!(b, c, "Clone preserves value");
}

#[test]
fn resource_handle_debug_round_trip() {
    let h = ResourceHandle(99);
    let s = format!("{:?}", h);
    assert!(s.contains("99"), "Debug output contains the inner value");
}

#[test]
fn resource_handle_display_non_none() {
    let h = ResourceHandle(1);
    let s = format!("{}", h);
    assert!(s.contains("1"), "Display contains handle value");
    assert!(!s.contains("NONE"), "Display does not say NONE for normal handles");
}

#[test]
fn resource_handle_display_none() {
    let s = format!("{}", ResourceHandle::NONE);
    assert!(s.contains("NONE"), "Display for NONE mentions sentinel");
}

#[test]
fn resource_handle_equality_and_ordering() {
    let low = ResourceHandle(0);
    let mid = ResourceHandle(500);
    let high = ResourceHandle(1000);

    assert_eq!(low, low, "Handle equals itself");
    assert_ne!(low, high, "Different handles are not equal");
    assert!(low < mid, "Handles ordered by inner value");
    assert!(mid < high, "Handles ordered by inner value");
    assert!(high > low, "Handles ordered descending");
}

#[test]
fn resource_handle_hash_consistency() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ResourceHandle(1));
    set.insert(ResourceHandle(2));
    set.insert(ResourceHandle(1)); // duplicate
    assert_eq!(set.len(), 2, "HashSet deduplicates by handle value");
}

// =============================================================================
// SECTION 2 -- PassIndex: construction, Display
// =============================================================================

#[test]
fn pass_index_constructs_with_raw_usize() {
    let p = PassIndex(3);
    assert_eq!(p.0, 3, "PassIndex wraps a usize");
}

#[test]
fn pass_index_copy_and_clone() {
    let a = PassIndex(10);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b, "Copy preserves value");
    assert_eq!(b, c, "Clone preserves value");
}

#[test]
fn pass_index_display() {
    let p = PassIndex(7);
    let s = format!("{}", p);
    assert!(s.contains("7"), "Display contains the pass index");
    assert!(s.contains("PassIndex"), "Display mentions the type");
}

#[test]
fn pass_index_ordering() {
    let early = PassIndex(0);
    let late = PassIndex(4);
    assert!(early < late, "Earlier passes have smaller indices");
    assert!(late > early, "Later passes have larger indices");
}

// =============================================================================
// SECTION 3 -- PassType: variant discrimination, Display
// =============================================================================

#[test]
fn pass_type_variants_discriminated() {
    // All four variants must be constructable.
    let g = PassType::Graphics;
    let c = PassType::Compute;
    let cp = PassType::Copy;
    let rt = PassType::RayTracing;

    // Verify structural equality.
    assert_eq!(g, PassType::Graphics);
    assert_eq!(c, PassType::Compute);
    assert_eq!(cp, PassType::Copy);
    assert_eq!(rt, PassType::RayTracing);

    // Verify they are distinct.
    assert_ne!(g, c);
    assert_ne!(g, cp);
    assert_ne!(g, rt);
    assert_ne!(c, cp);
    assert_ne!(c, rt);
    assert_ne!(cp, rt);
}

#[test]
fn pass_type_display_all_variants() {
    assert_eq!(format!("{}", PassType::Graphics), "Graphics");
    assert_eq!(format!("{}", PassType::Compute), "Compute");
    assert_eq!(format!("{}", PassType::Copy), "Copy");
    assert_eq!(format!("{}", PassType::RayTracing), "RayTracing");
}

#[test]
fn pass_type_clone_and_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(PassType::Compute);
    set.insert(PassType::Compute); // duplicate
    assert_eq!(set.len(), 1, "PassType is hashable");
    assert_eq!(
        PassType::Graphics.clone(),
        PassType::Graphics,
        "PassType clonable"
    );
}

// =============================================================================
// SECTION 4 -- ResourceAccess: variant discrimination, Display
// =============================================================================

#[test]
fn resource_access_variants_discriminated() {
    assert_eq!(ResourceAccess::Read, ResourceAccess::Read);
    assert_ne!(ResourceAccess::Read, ResourceAccess::Write);
    assert_ne!(ResourceAccess::Write, ResourceAccess::ReadWrite);
    assert_ne!(ResourceAccess::Read, ResourceAccess::ReadWrite);
}

#[test]
fn resource_access_display() {
    assert_eq!(format!("{}", ResourceAccess::Read), "Read");
    assert_eq!(format!("{}", ResourceAccess::Write), "Write");
    assert_eq!(format!("{}", ResourceAccess::ReadWrite), "ReadWrite");
}

#[test]
fn resource_access_clone_and_copy() {
    let a = ResourceAccess::ReadWrite;
    let b = a;
    let c = a.clone();
    assert_eq!(a, b, "Copy works");
    assert_eq!(b, c, "Clone works");
}

// =============================================================================
// SECTION 5 -- ResourceAccessEntry: construction, Display
// =============================================================================

#[test]
fn resource_access_entry_new_constructs_pair() {
    let entry = ResourceAccessEntry::new(ResourceHandle(3), ResourceAccess::Write);
    assert_eq!(
        entry.resource, ResourceHandle(3),
        "Entry stores resource handle"
    );
    assert_eq!(
        entry.access, ResourceAccess::Write,
        "Entry stores access type"
    );
}

#[test]
fn resource_access_entry_display_read() {
    let entry = ResourceAccessEntry::new(ResourceHandle(10), ResourceAccess::Read);
    let s = format!("{}", entry);
    assert!(s.contains("Read"), "Display shows Read");
    assert!(s.contains("10"), "Display shows handle");
}

#[test]
fn resource_access_entry_display_read_write() {
    let entry = ResourceAccessEntry::new(ResourceHandle(5), ResourceAccess::ReadWrite);
    let s = format!("{}", entry);
    assert!(s.contains("ReadWrite"), "Display shows ReadWrite");
}

// =============================================================================
// SECTION 6 -- ResourceAccessSet: empty, is_empty, len, contains, Display
// =============================================================================

#[test]
fn resource_access_set_empty_creates_empty_set() {
    let set = ResourceAccessSet::empty();
    assert!(set.reads.is_empty(), "Empty set has no reads");
    assert!(set.writes.is_empty(), "Empty set has no writes");
}

#[test]
fn resource_access_set_is_empty_true_when_no_entries() {
    let set = ResourceAccessSet::empty();
    assert!(set.is_empty(), "Default-constructed set is empty");
}

#[test]
fn resource_access_set_is_empty_false_when_reads_present() {
    let mut set = ResourceAccessSet::empty();
    set.reads.push(ResourceHandle(1));
    assert!(!set.is_empty(), "Set with reads is not empty");
}

#[test]
fn resource_access_set_is_empty_false_when_writes_present() {
    let mut set = ResourceAccessSet::empty();
    set.writes.push(ResourceHandle(2));
    assert!(!set.is_empty(), "Set with writes is not empty");
}

#[test]
fn resource_access_set_len_counts_reads_plus_writes() {
    let mut set = ResourceAccessSet::empty();
    set.reads.push(ResourceHandle(1));
    set.reads.push(ResourceHandle(2));
    set.writes.push(ResourceHandle(3));
    assert_eq!(set.len(), 3, "len() = reads + writes");
}

#[test]
fn resource_access_set_len_zero_for_empty() {
    let set = ResourceAccessSet::empty();
    assert_eq!(set.len(), 0, "Empty set len is 0");
}

#[test]
fn resource_access_set_contains_read_handle() {
    let mut set = ResourceAccessSet::empty();
    set.reads.push(ResourceHandle(7));
    assert!(set.contains(ResourceHandle(7)), "Finds handle in reads");
}

#[test]
fn resource_access_set_contains_write_handle() {
    let mut set = ResourceAccessSet::empty();
    set.writes.push(ResourceHandle(8));
    assert!(set.contains(ResourceHandle(8)), "Finds handle in writes");
}

#[test]
fn resource_access_set_does_not_contain_missing_handle() {
    let set = ResourceAccessSet::empty();
    assert!(!set.contains(ResourceHandle(99)), "Missing handle not found");
}

#[test]
fn resource_access_set_default_is_empty() {
    let set = ResourceAccessSet::default();
    assert!(set.is_empty(), "Default trait gives empty set");
}

#[test]
fn resource_access_set_display_empty() {
    let set = ResourceAccessSet::empty();
    let s = format!("{}", set);
    assert!(s.contains("reads:["), "Display shows reads header");
    assert!(s.contains("writes:["), "Display shows writes header");
}

#[test]
fn resource_access_set_display_non_empty() {
    let mut set = ResourceAccessSet::empty();
    set.reads.push(ResourceHandle(1));
    set.reads.push(ResourceHandle(2));
    set.writes.push(ResourceHandle(3));
    let s = format!("{}", set);
    assert!(s.contains("1"), "Display contains read handle 1");
    assert!(s.contains("2"), "Display contains read handle 2");
    assert!(s.contains("3"), "Display contains write handle 3");
}

// =============================================================================
// SECTION 7 -- AttachmentLoadOp / AttachmentStoreOp: variants, Display
// =============================================================================

#[test]
fn attachment_load_op_variants() {
    assert_eq!(AttachmentLoadOp::Load, AttachmentLoadOp::Load);
    assert_eq!(AttachmentLoadOp::Clear, AttachmentLoadOp::Clear);
    assert_eq!(AttachmentLoadOp::DontCare, AttachmentLoadOp::DontCare);
    assert_ne!(AttachmentLoadOp::Load, AttachmentLoadOp::Clear);
    assert_ne!(AttachmentLoadOp::Clear, AttachmentLoadOp::DontCare);
}

#[test]
fn attachment_load_op_display() {
    assert_eq!(format!("{}", AttachmentLoadOp::Load), "Load");
    assert_eq!(format!("{}", AttachmentLoadOp::Clear), "Clear");
    assert_eq!(format!("{}", AttachmentLoadOp::DontCare), "DontCare");
}

#[test]
fn attachment_store_op_variants() {
    assert_eq!(AttachmentStoreOp::Store, AttachmentStoreOp::Store);
    assert_eq!(AttachmentStoreOp::DontCare, AttachmentStoreOp::DontCare);
    assert_ne!(AttachmentStoreOp::Store, AttachmentStoreOp::DontCare);
}

#[test]
fn attachment_store_op_display() {
    assert_eq!(format!("{}", AttachmentStoreOp::Store), "Store");
    assert_eq!(format!("{}", AttachmentStoreOp::DontCare), "DontCare");
}

// =============================================================================
// SECTION 8 -- ColorAttachment: full field construction, default, Display
// =============================================================================

#[test]
fn color_attachment_default_fields() {
    let att = ColorAttachment::default();
    assert_eq!(att.resource, ResourceHandle::NONE, "Default resource is NONE");
    assert_eq!(att.mip_level, 0, "Default mip_level is 0");
    assert_eq!(att.array_layer, 0, "Default array_layer is 0");
    assert_eq!(att.load_op, AttachmentLoadOp::Load, "Default load_op is Load");
    assert_eq!(att.store_op, AttachmentStoreOp::Store, "Default store_op is Store");
    assert_eq!(att.clear_color, [0.0, 0.0, 0.0, 0.0], "Default clear_color is zero");
}

#[test]
fn color_attachment_full_field_construction() {
    let att = ColorAttachment {
        resource: ResourceHandle(5),
        mip_level: 2,
        array_layer: 1,
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        clear_color: [0.1, 0.2, 0.3, 1.0],
    };

    assert_eq!(att.resource, ResourceHandle(5));
    assert_eq!(att.mip_level, 2);
    assert_eq!(att.array_layer, 1);
    assert_eq!(att.load_op, AttachmentLoadOp::Clear);
    assert_eq!(att.store_op, AttachmentStoreOp::Store);
    assert_eq!(att.clear_color, [0.1, 0.2, 0.3, 1.0]);
}

#[test]
fn color_attachment_discard_store_op() {
    let att = ColorAttachment {
        resource: ResourceHandle(1),
        load_op: AttachmentLoadOp::DontCare,
        store_op: AttachmentStoreOp::DontCare,
        ..Default::default()
    };
    assert_eq!(att.load_op, AttachmentLoadOp::DontCare, "DontCare load");
    assert_eq!(att.store_op, AttachmentStoreOp::DontCare, "DontCare store");
}

#[test]
fn color_attachment_mip_and_array_layer_varied() {
    let att = ColorAttachment {
        resource: ResourceHandle(2),
        mip_level: 4,
        array_layer: 3,
        ..Default::default()
    };
    assert_eq!(att.mip_level, 4, "Mip level 4");
    assert_eq!(att.array_layer, 3, "Array layer 3");
}

#[test]
fn color_attachment_display() {
    let att = ColorAttachment {
        resource: ResourceHandle(10),
        mip_level: 1,
        array_layer: 0,
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        clear_color: [0.5, 0.5, 0.5, 1.0],
    };
    let s = format!("{}", att);
    assert!(s.contains("10"), "Display contains resource handle");
    assert!(s.contains("Clear"), "Display shows Clear load op");
    assert!(s.contains("Store"), "Display shows Store store op");
}

// =============================================================================
// SECTION 9 -- DepthStencilAttachment: full field, default, Display
// =============================================================================

#[test]
fn depth_stencil_attachment_default_fields() {
    let ds = DepthStencilAttachment::default();
    assert_eq!(ds.resource, ResourceHandle::NONE);
    assert_eq!(ds.depth_load_op, AttachmentLoadOp::Load);
    assert_eq!(ds.depth_store_op, AttachmentStoreOp::Store);
    assert_eq!(ds.stencil_load_op, AttachmentLoadOp::Load);
    assert_eq!(ds.stencil_store_op, AttachmentStoreOp::DontCare);
    assert_eq!(ds.clear_depth, 1.0, "Default clear_depth is 1.0");
    assert_eq!(ds.clear_stencil, 0, "Default clear_stencil is 0");
    assert!(ds.depth_test_enabled, "Depth test enabled by default");
    assert!(ds.depth_write_enabled, "Depth write enabled by default");
}

#[test]
fn depth_stencil_attachment_full_field_construction() {
    let ds = DepthStencilAttachment {
        resource: ResourceHandle(3),
        depth_load_op: AttachmentLoadOp::Clear,
        depth_store_op: AttachmentStoreOp::Store,
        stencil_load_op: AttachmentLoadOp::DontCare,
        stencil_store_op: AttachmentStoreOp::DontCare,
        clear_depth: 0.5,
        clear_stencil: 0xFF,
        depth_test_enabled: true,
        depth_write_enabled: false,
    };

    assert_eq!(ds.resource, ResourceHandle(3));
    assert_eq!(ds.depth_load_op, AttachmentLoadOp::Clear);
    assert_eq!(ds.depth_store_op, AttachmentStoreOp::Store);
    assert_eq!(ds.stencil_load_op, AttachmentLoadOp::DontCare);
    assert_eq!(ds.stencil_store_op, AttachmentStoreOp::DontCare);
    assert_eq!(ds.clear_depth, 0.5);
    assert_eq!(ds.clear_stencil, 0xFF);
    assert!(ds.depth_test_enabled);
    assert!(!ds.depth_write_enabled);
}

#[test]
fn depth_stencil_attachment_depth_write_disabled() {
    let ds = DepthStencilAttachment {
        depth_write_enabled: false,
        ..Default::default()
    };
    assert!(!ds.depth_write_enabled, "Depth writes can be disabled");
}

#[test]
fn depth_stencil_attachment_stencil_only_clear() {
    let ds = DepthStencilAttachment {
        depth_load_op: AttachmentLoadOp::Load,
        stencil_load_op: AttachmentLoadOp::Clear,
        clear_stencil: 42,
        ..Default::default()
    };
    assert_eq!(ds.depth_load_op, AttachmentLoadOp::Load);
    assert_eq!(ds.stencil_load_op, AttachmentLoadOp::Clear);
    assert_eq!(ds.clear_stencil, 42);
}

#[test]
fn depth_stencil_attachment_display() {
    let ds = DepthStencilAttachment {
        resource: ResourceHandle(7),
        ..Default::default()
    };
    let s = format!("{}", ds);
    assert!(s.contains("7"), "Display contains resource handle");
    assert!(s.contains("true"), "Display shows boolean flags");
    assert!(s.contains("DepthStencilAttachment"), "Display shows type name");
}

// =============================================================================
// SECTION 10 -- InstanceSource: Direct/Indirect/Mesh, Display
// =============================================================================

#[test]
fn instance_source_direct_fields() {
    let src = InstanceSource::Direct {
        index_count: 36,
        instance_count: 2,
        base_vertex: 0,
        first_index: 0,
        first_instance: 0,
    };

    if let InstanceSource::Direct {
        index_count,
        instance_count,
        base_vertex,
        first_index,
        first_instance,
    } = src
    {
        assert_eq!(index_count, 36);
        assert_eq!(instance_count, 2);
        assert_eq!(base_vertex, 0);
        assert_eq!(first_index, 0);
        assert_eq!(first_instance, 0);
    } else {
        panic!("Expected Direct variant");
    }
}

#[test]
fn instance_source_direct_non_zero_base_vertex() {
    let src = InstanceSource::Direct {
        index_count: 24,
        instance_count: 1,
        base_vertex: 100,
        first_index: 12,
        first_instance: 3,
    };

    if let InstanceSource::Direct {
        base_vertex,
        first_index,
        first_instance,
        ..
    } = src
    {
        assert_eq!(base_vertex, 100);
        assert_eq!(first_index, 12);
        assert_eq!(first_instance, 3);
    } else {
        panic!("Expected Direct variant");
    }
}

#[test]
fn instance_source_indirect_fields() {
    let src = InstanceSource::Indirect {
        buffer: ResourceHandle(10),
        offset: 256,
        draw_count: 8,
        stride: 20,
    };

    if let InstanceSource::Indirect {
        buffer,
        offset,
        draw_count,
        stride,
    } = src
    {
        assert_eq!(buffer, ResourceHandle(10));
        assert_eq!(offset, 256);
        assert_eq!(draw_count, 8);
        assert_eq!(stride, 20);
    } else {
        panic!("Expected Indirect variant");
    }
}

#[test]
fn instance_source_mesh_fields() {
    let src = InstanceSource::Mesh {
        group_count_x: 16,
        group_count_y: 8,
        group_count_z: 1,
    };

    if let InstanceSource::Mesh {
        group_count_x,
        group_count_y,
        group_count_z,
    } = src
    {
        assert_eq!(group_count_x, 16);
        assert_eq!(group_count_y, 8);
        assert_eq!(group_count_z, 1);
    } else {
        panic!("Expected Mesh variant");
    }
}

#[test]
fn instance_source_direct_display() {
    let src = InstanceSource::Direct {
        index_count: 42,
        instance_count: 3,
        base_vertex: 0,
        first_index: 0,
        first_instance: 0,
    };
    let s = format!("{}", src);
    assert!(s.contains("Direct"), "Display shows variant name");
    assert!(s.contains("42"), "Display shows index count");
}

#[test]
fn instance_source_indirect_display() {
    let src = InstanceSource::Indirect {
        buffer: ResourceHandle(5),
        offset: 128,
        draw_count: 4,
        stride: 32,
    };
    let s = format!("{}", src);
    assert!(s.contains("Indirect"));
    assert!(s.contains("5"));
}

#[test]
fn instance_source_mesh_display() {
    let src = InstanceSource::Mesh {
        group_count_x: 32,
        group_count_y: 16,
        group_count_z: 8,
    };
    let s = format!("{}", src);
    assert!(s.contains("Mesh"));
    assert!(s.contains("32"));
    assert!(s.contains("16"));
    assert!(s.contains("8"));
}

#[test]
fn instance_source_variants_distinct() {
    let direct = InstanceSource::Direct {
        index_count: 6,
        instance_count: 1,
        base_vertex: 0,
        first_index: 0,
        first_instance: 0,
    };
    let indirect = InstanceSource::Indirect {
        buffer: ResourceHandle(0),
        offset: 0,
        draw_count: 1,
        stride: 0,
    };
    let mesh = InstanceSource::Mesh {
        group_count_x: 1,
        group_count_y: 1,
        group_count_z: 1,
    };
    assert_ne!(direct, indirect, "Direct != Indirect");
    assert_ne!(direct, mesh, "Direct != Mesh");
    assert_ne!(indirect, mesh, "Indirect != Mesh");
}

// =============================================================================
// SECTION 11 -- DispatchSource: Direct/Indirect, Display
// =============================================================================

#[test]
fn dispatch_source_direct_fields() {
    let src = DispatchSource::Direct {
        group_count_x: 8,
        group_count_y: 4,
        group_count_z: 1,
    };

    if let DispatchSource::Direct {
        group_count_x,
        group_count_y,
        group_count_z,
    } = src
    {
        assert_eq!(group_count_x, 8);
        assert_eq!(group_count_y, 4);
        assert_eq!(group_count_z, 1);
    } else {
        panic!("Expected Direct variant");
    }
}

#[test]
fn dispatch_source_indirect_fields() {
    let src = DispatchSource::Indirect {
        buffer: ResourceHandle(12),
        offset: 64,
    };

    if let DispatchSource::Indirect { buffer, offset } = src {
        assert_eq!(buffer, ResourceHandle(12));
        assert_eq!(offset, 64);
    } else {
        panic!("Expected Indirect variant");
    }
}

#[test]
fn dispatch_source_direct_large_grid() {
    let src = DispatchSource::Direct {
        group_count_x: 64,
        group_count_y: 64,
        group_count_z: 1,
    };
    if let DispatchSource::Direct {
        group_count_x,
        group_count_y,
        ..
    } = src
    {
        assert_eq!(group_count_x, 64, "Large X workgroup count");
        assert_eq!(group_count_y, 64, "Large Y workgroup count");
    } else {
        panic!("Expected Direct variant");
    }
}

#[test]
fn dispatch_source_direct_display() {
    let src = DispatchSource::Direct {
        group_count_x: 16,
        group_count_y: 8,
        group_count_z: 4,
    };
    let s = format!("{}", src);
    assert!(s.contains("Direct"));
    assert!(s.contains("16"));
    assert!(s.contains("8"));
    assert!(s.contains("4"));
}

#[test]
fn dispatch_source_indirect_display() {
    let src = DispatchSource::Indirect {
        buffer: ResourceHandle(9),
        offset: 256,
    };
    let s = format!("{}", src);
    assert!(s.contains("Indirect"));
    assert!(s.contains("9"));
    assert!(s.contains("256"));
}

#[test]
fn dispatch_source_variants_distinct() {
    let direct = DispatchSource::Direct {
        group_count_x: 1,
        group_count_y: 1,
        group_count_z: 1,
    };
    let indirect = DispatchSource::Indirect {
        buffer: ResourceHandle(0),
        offset: 0,
    };
    assert_ne!(direct, indirect, "Direct != Indirect");
}

// =============================================================================
// SECTION 12 -- ViewType: variant discrimination, Display
// =============================================================================

#[test]
fn view_type_all_variants_accessible() {
    let _ = ViewType::Texture2D;
    let _ = ViewType::TextureCube;
    let _ = ViewType::Texture3D;
    let _ = ViewType::Storage;
    let _ = ViewType::UniformTexel;
    let _ = ViewType::StorageTexel;
    let _ = ViewType::UniformBuffer;
    let _ = ViewType::StorageBuffer;
    let _ = ViewType::AccelerationStructure;
}

#[test]
fn view_type_variants_distinct() {
    assert_ne!(ViewType::Texture2D, ViewType::Storage);
    assert_ne!(ViewType::UniformBuffer, ViewType::StorageBuffer);
    assert_ne!(ViewType::Texture2D, ViewType::TextureCube);
    assert_ne!(ViewType::Texture3D, ViewType::TextureCube);
    assert_ne!(ViewType::UniformTexel, ViewType::StorageTexel);
}

#[test]
fn view_type_display_all_variants() {
    assert_eq!(format!("{}", ViewType::Texture2D), "Texture2D");
    assert_eq!(format!("{}", ViewType::TextureCube), "TextureCube");
    assert_eq!(format!("{}", ViewType::Texture3D), "Texture3D");
    assert_eq!(format!("{}", ViewType::Storage), "Storage");
    assert_eq!(format!("{}", ViewType::UniformTexel), "UniformTexel");
    assert_eq!(format!("{}", ViewType::StorageTexel), "StorageTexel");
    assert_eq!(format!("{}", ViewType::UniformBuffer), "UniformBuffer");
    assert_eq!(format!("{}", ViewType::StorageBuffer), "StorageBuffer");
    assert_eq!(
        format!("{}", ViewType::AccelerationStructure),
        "AccelerationStructure"
    );
}

// =============================================================================
// SECTION 13 -- TextureDesc / Texture3DDesc / BufferDesc: Display
// =============================================================================

#[test]
fn texture_desc_standard_resolution() {
    let desc = TextureDesc {
        width: 1920,
        height: 1080,
        mip_levels: 1,
        array_layers: 1,
        format: "rgba8unorm".into(),
    };
    assert_eq!(desc.width, 1920);
    assert_eq!(desc.height, 1080);
    assert_eq!(desc.mip_levels, 1);
    assert_eq!(desc.array_layers, 1);
    assert_eq!(desc.format, "rgba8unorm");
}

#[test]
fn texture_desc_multi_mip_and_array() {
    let desc = TextureDesc {
        width: 512,
        height: 512,
        mip_levels: 10,
        array_layers: 6,
        format: "bc7_unorm".into(),
    };
    assert_eq!(desc.mip_levels, 10, "Full mip chain");
    assert_eq!(desc.array_layers, 6, "6 array layers (e.g. cubemap)");
}

#[test]
fn texture_desc_display() {
    let desc = TextureDesc {
        width: 256,
        height: 256,
        mip_levels: 8,
        array_layers: 1,
        format: "r32float".into(),
    };
    let s = format!("{}", desc);
    assert!(s.contains("256"), "Display contains width/height");
    assert!(s.contains("8"), "Display contains mip level count");
    assert!(s.contains("r32float"), "Display contains format");
}

#[test]
fn texture_3d_desc_full_field() {
    let desc = Texture3DDesc {
        width: 128,
        height: 128,
        depth: 128,
        mip_levels: 7,
        format: "r16float".into(),
    };
    assert_eq!(desc.width, 128);
    assert_eq!(desc.height, 128);
    assert_eq!(desc.depth, 128);
    assert_eq!(desc.mip_levels, 7);
    assert_eq!(desc.format, "r16float");
}

#[test]
fn texture_3d_desc_display() {
    let desc = Texture3DDesc {
        width: 64,
        height: 64,
        depth: 64,
        mip_levels: 6,
        format: "rgba16float".into(),
    };
    let s = format!("{}", desc);
    assert!(s.contains("64"), "Display contains dimensions");
    assert!(s.contains("rgba16float"), "Display contains format");
}

#[test]
fn buffer_desc_full_field() {
    let desc = BufferDesc {
        size: 1048576,
        usage: "storage | indirect".into(),
        is_indirect_arg: true,
    };
    assert_eq!(desc.size, 1048576);
    assert_eq!(desc.usage, "storage | indirect");
    assert!(desc.is_indirect_arg);
}

#[test]
fn buffer_desc_non_indirect() {
    let desc = BufferDesc {
        size: 4096,
        usage: "uniform".into(),
        is_indirect_arg: false,
    };
    assert_eq!(desc.size, 4096);
    assert!(!desc.is_indirect_arg, "Buffer is not an indirect arg");
}

#[test]
fn buffer_desc_display() {
    let desc = BufferDesc {
        size: 65536,
        usage: "storage".into(),
        is_indirect_arg: false,
    };
    let s = format!("{}", desc);
    assert!(s.contains("65536"), "Display contains size");
    assert!(s.contains("storage"), "Display contains usage");
}

// =============================================================================
// SECTION 14 -- ResourceDesc: variant discrimination, Display
// =============================================================================

#[test]
fn resource_desc_texture_2d_variant() {
    let desc = ResourceDesc::Texture2D(TextureDesc {
        width: 1920,
        height: 1080,
        mip_levels: 1,
        array_layers: 1,
        format: "rgba8unorm".into(),
    });

    match &desc {
        ResourceDesc::Texture2D(t) => {
            assert_eq!(t.width, 1920);
            assert_eq!(t.format, "rgba8unorm");
        }
        _ => panic!("Expected Texture2D variant"),
    }
}

#[test]
fn resource_desc_texture_3d_variant() {
    let desc = ResourceDesc::Texture3D(Texture3DDesc {
        width: 256,
        height: 256,
        depth: 256,
        mip_levels: 1,
        format: "rgba16float".into(),
    });

    match &desc {
        ResourceDesc::Texture3D(t) => {
            assert_eq!(t.depth, 256);
        }
        _ => panic!("Expected Texture3D variant"),
    }
}

#[test]
fn resource_desc_texture_cube_variant() {
    let desc = ResourceDesc::TextureCube(TextureDesc {
        width: 1024,
        height: 1024,
        mip_levels: 10,
        array_layers: 6,
        format: "bc7_unorm".into(),
    });

    match &desc {
        ResourceDesc::TextureCube(t) => {
            assert_eq!(t.array_layers, 6, "Cubemap has 6 layers");
        }
        _ => panic!("Expected TextureCube variant"),
    }
}

#[test]
fn resource_desc_buffer_variant() {
    let desc = ResourceDesc::Buffer(BufferDesc {
        size: 262144,
        usage: "storage".into(),
        is_indirect_arg: true,
    });

    match &desc {
        ResourceDesc::Buffer(b) => {
            assert_eq!(b.size, 262144);
            assert!(b.is_indirect_arg);
        }
        _ => panic!("Expected Buffer variant"),
    }
}

#[test]
fn resource_desc_display_all_variants() {
    let t2 = ResourceDesc::Texture2D(TextureDesc {
        width: 4,
        height: 4,
        mip_levels: 1,
        array_layers: 1,
        format: "r8unorm".into(),
    });
    assert!(format!("{}", t2).contains("Texture2D"));

    let t3 = ResourceDesc::Texture3D(Texture3DDesc {
        width: 4,
        height: 4,
        depth: 4,
        mip_levels: 1,
        format: "r8unorm".into(),
    });
    assert!(format!("{}", t3).contains("Texture3D"));

    let tc = ResourceDesc::TextureCube(TextureDesc {
        width: 4,
        height: 4,
        mip_levels: 1,
        array_layers: 6,
        format: "r8unorm".into(),
    });
    assert!(format!("{}", tc).contains("TextureCube"));

    let buf = ResourceDesc::Buffer(BufferDesc {
        size: 64,
        usage: "uniform".into(),
        is_indirect_arg: false,
    });
    assert!(format!("{}", buf).contains("Buffer"));
}

// =============================================================================
// SECTION 15 -- ResourceLifetime / ResourceState: Display
// =============================================================================

#[test]
fn resource_lifetime_variants_and_display() {
    assert_eq!(ResourceLifetime::Transient, ResourceLifetime::Transient);
    assert_eq!(ResourceLifetime::Imported, ResourceLifetime::Imported);
    assert_ne!(ResourceLifetime::Transient, ResourceLifetime::Imported);
    assert_eq!(format!("{}", ResourceLifetime::Transient), "Transient");
    assert_eq!(format!("{}", ResourceLifetime::Imported), "Imported");
}

#[test]
fn resource_state_all_variants_display() {
    assert_eq!(format!("{}", ResourceState::Uninitialized), "Uninitialized");
    assert_eq!(format!("{}", ResourceState::VertexBuffer), "VertexBuffer");
    assert_eq!(format!("{}", ResourceState::IndexBuffer), "IndexBuffer");
    assert_eq!(format!("{}", ResourceState::IndirectArgument), "IndirectArgument");
    assert_eq!(format!("{}", ResourceState::ColorAttachment), "ColorAttachment");
    assert_eq!(format!("{}", ResourceState::DepthStencilAttachment), "DepthStencilAttachment");
    assert_eq!(format!("{}", ResourceState::DepthStencilReadOnly), "DepthStencilReadOnly");
    assert_eq!(format!("{}", ResourceState::ShaderRead), "ShaderRead");
    assert_eq!(format!("{}", ResourceState::ShaderReadWrite), "ShaderReadWrite");
    assert_eq!(format!("{}", ResourceState::TransferSrc), "TransferSrc");
    assert_eq!(format!("{}", ResourceState::TransferDst), "TransferDst");
    assert_eq!(format!("{}", ResourceState::AccelerationStructure), "AccelerationStructure");
    assert_eq!(format!("{}", ResourceState::Present), "Present");
}

#[test]
fn resource_state_variants_distinct() {
    assert_ne!(ResourceState::Uninitialized, ResourceState::ShaderRead);
    assert_ne!(ResourceState::ShaderRead, ResourceState::ShaderReadWrite);
    assert_ne!(ResourceState::ColorAttachment, ResourceState::DepthStencilAttachment);
    assert_ne!(ResourceState::TransferSrc, ResourceState::TransferDst);
    assert_ne!(ResourceState::DepthStencilAttachment, ResourceState::DepthStencilReadOnly);
}

// =============================================================================
// SECTION 16 -- IrResource: new(), all fields, Display, view_format_override
// =============================================================================

#[test]
fn ir_resource_new_constructs_with_all_required_fields() {
    let res = IrResource::new(
        ResourceHandle(1),
        "gbuffer_albedo",
        ResourceDesc::Texture2D(TextureDesc {
            width: 1920,
            height: 1080,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    assert_eq!(res.handle, ResourceHandle(1), "Handle is preserved");
    assert_eq!(res.name, "gbuffer_albedo", "Name is preserved");
    assert_eq!(res.lifetime, ResourceLifetime::Transient, "Lifetime is set");
    assert_eq!(
        res.initial_state, ResourceState::Uninitialized,
        "Initial state is Uninitialized for transient resources"
    );
    assert!(res.view_format_override.is_none(), "No format override by default");
}

#[test]
fn ir_resource_imported_lifetime() {
    let res = IrResource::new(
        ResourceHandle(2),
        "swapchain_backbuffer",
        ResourceDesc::Texture2D(TextureDesc {
            width: 1920,
            height: 1080,
            mip_levels: 1,
            array_layers: 1,
            format: "bgra8unorm-srgb".into(),
        }),
        ResourceLifetime::Imported,
        ResourceState::Present,
    );

    assert_eq!(res.lifetime, ResourceLifetime::Imported);
    assert_eq!(
        res.initial_state, ResourceState::Present,
        "Imported resources start in Present state"
    );
}

#[test]
fn ir_resource_buffer_type() {
    let res = IrResource::new(
        ResourceHandle(3),
        "indirect_buffer",
        ResourceDesc::Buffer(BufferDesc {
            size: 4096,
            usage: "indirect".into(),
            is_indirect_arg: true,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    match &res.desc {
        ResourceDesc::Buffer(b) => {
            assert!(b.is_indirect_arg, "Is an indirect argument buffer");
            assert_eq!(b.size, 4096);
        }
        other => panic!("Expected Buffer desc, got {:?}", other),
    }
}

#[test]
fn ir_resource_view_format_override() {
    let mut res = IrResource::new(
        ResourceHandle(4),
        "srgb_override",
        ResourceDesc::Texture2D(TextureDesc {
            width: 800,
            height: 600,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Imported,
        ResourceState::ShaderRead,
    );

    assert!(res.view_format_override.is_none(), "Initially no override");
    res.view_format_override = Some("bgra8unorm-srgb".into());
    assert_eq!(
        res.view_format_override,
        Some("bgra8unorm-srgb".into()),
        "View format override can be set"
    );
}

#[test]
fn ir_resource_cube_map() {
    let res = IrResource::new(
        ResourceHandle(5),
        "env_map",
        ResourceDesc::TextureCube(TextureDesc {
            width: 1024,
            height: 1024,
            mip_levels: 10,
            array_layers: 6,
            format: "bc7_unorm".into(),
        }),
        ResourceLifetime::Imported,
        ResourceState::ShaderRead,
    );

    assert_eq!(res.name, "env_map");
    match &res.desc {
        ResourceDesc::TextureCube(t) => {
            assert_eq!(t.array_layers, 6, "Cubemap has 6 layers");
        }
        _ => panic!("Expected TextureCube"),
    }
}

#[test]
fn ir_resource_3d_texture() {
    let res = IrResource::new(
        ResourceHandle(6),
        "volume_data",
        ResourceDesc::Texture3D(Texture3DDesc {
            width: 256,
            height: 256,
            depth: 256,
            mip_levels: 1,
            format: "r16float".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    match &res.desc {
        ResourceDesc::Texture3D(t) => {
            assert_eq!(t.depth, 256, "3D texture depth");
        }
        _ => panic!("Expected Texture3D"),
    }
}

#[test]
fn ir_resource_name_is_preserved_exactly() {
    let res = IrResource::new(
        ResourceHandle(10),
        "custom_resource_name_123",
        ResourceDesc::Texture2D(TextureDesc {
            width: 1,
            height: 1,
            mip_levels: 1,
            array_layers: 1,
            format: "r8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    assert_eq!(res.name, "custom_resource_name_123");
}

#[test]
fn ir_resource_display() {
    let res = IrResource::new(
        ResourceHandle(42),
        "test_resource",
        ResourceDesc::Texture2D(TextureDesc {
            width: 512,
            height: 512,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );
    let s = format!("{}", res);
    assert!(s.contains("42"), "Display contains handle");
    assert!(s.contains("test_resource"), "Display contains name");
    assert!(s.contains("Transient"), "Display contains lifetime");
}

// =============================================================================
// SECTION 17 -- IrPass: constructors (graphics/compute/copy/ray_tracing)
// =============================================================================

#[test]
fn ir_pass_graphics_constructor_sets_all_required_fields() {
    let color_att = ColorAttachment {
        resource: ResourceHandle(10),
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        ..Default::default()
    };
    let ds = DepthStencilAttachment {
        resource: ResourceHandle(11),
        ..Default::default()
    };

    let pass = IrPass::graphics(
        PassIndex(0),
        "gbuffer",
        vec![color_att],
        Some(ds),
        InstanceSource::Direct {
            index_count: 36,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    assert_eq!(pass.index, PassIndex(0));
    assert_eq!(pass.name, "gbuffer");
    assert_eq!(pass.pass_type, PassType::Graphics);
    assert_eq!(pass.color_attachments.len(), 1);
    assert!(pass.depth_stencil.is_some());
    assert_eq!(pass.view_type, ViewType::Texture2D);
    assert!(pass.tags.is_empty(), "No tags by default");
}

#[test]
fn ir_pass_graphics_multiple_color_attachments() {
    let ca1 = ColorAttachment {
        resource: ResourceHandle(1),
        load_op: AttachmentLoadOp::Clear,
        ..Default::default()
    };
    let ca2 = ColorAttachment {
        resource: ResourceHandle(2),
        load_op: AttachmentLoadOp::Clear,
        ..Default::default()
    };
    let ca3 = ColorAttachment {
        resource: ResourceHandle(3),
        load_op: AttachmentLoadOp::Clear,
        ..Default::default()
    };
    let ca4 = ColorAttachment {
        resource: ResourceHandle(4),
        load_op: AttachmentLoadOp::Load,
        ..Default::default()
    };

    let pass = IrPass::graphics(
        PassIndex(0),
        "mrt",
        vec![ca1, ca2, ca3, ca4],
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

    assert_eq!(pass.color_attachments.len(), 4, "MRT with 4 attachments");
    assert!(
        pass.depth_stencil.is_none(),
        "No depth-stencil for MRT-only pass"
    );
}

#[test]
fn ir_pass_graphics_without_depth_stencil() {
    let ca = ColorAttachment {
        resource: ResourceHandle(1),
        ..Default::default()
    };
    let pass = IrPass::graphics(
        PassIndex(0),
        "no_depth",
        vec![ca],
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

    assert!(pass.depth_stencil.is_none(), "Graphics pass without depth");
}

#[test]
fn ir_pass_compute_constructor() {
    let pass = IrPass::compute(
        PassIndex(1),
        "postfx_bloom",
        DispatchSource::Direct {
            group_count_x: 16,
            group_count_y: 16,
            group_count_z: 1,
        },
        ViewType::Storage,
    );

    assert_eq!(pass.index, PassIndex(1));
    assert_eq!(pass.name, "postfx_bloom");
    assert_eq!(pass.pass_type, PassType::Compute);
    assert!(pass.color_attachments.is_empty(), "Compute has no colour attachments");
    assert!(pass.depth_stencil.is_none(), "Compute has no depth-stencil");
    assert_eq!(pass.view_type, ViewType::Storage);
    assert!(pass.dispatch_source.is_some(), "Compute has a dispatch source");
    assert!(pass.tags.is_empty());
}

#[test]
fn ir_pass_compute_indirect_dispatch() {
    let pass = IrPass::compute(
        PassIndex(2),
        "indirect_comp",
        DispatchSource::Indirect {
            buffer: ResourceHandle(20),
            offset: 0,
        },
        ViewType::StorageBuffer,
    );

    match pass.dispatch_source {
        Some(DispatchSource::Indirect { buffer, .. }) => {
            assert_eq!(buffer, ResourceHandle(20));
        }
        _ => panic!("Expected Indirect dispatch"),
    }
}

#[test]
fn ir_pass_copy_constructor() {
    let pass = IrPass::copy(PassIndex(3), "depth_copy");

    assert_eq!(pass.index, PassIndex(3));
    assert_eq!(pass.name, "depth_copy");
    assert_eq!(pass.pass_type, PassType::Copy);
    assert!(pass.color_attachments.is_empty(), "Copy has no colour attachments");
    assert!(pass.depth_stencil.is_none(), "Copy has no depth-stencil");
    assert!(pass.dispatch_source.is_none(), "Copy has no dispatch source");
    assert_eq!(pass.view_type, ViewType::StorageBuffer);
}

#[test]
fn ir_pass_ray_tracing_constructor() {
    let pass = IrPass::ray_tracing(
        PassIndex(4),
        "raytrace_gi",
        DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 8,
            group_count_z: 1,
        },
    );

    assert_eq!(pass.index, PassIndex(4));
    assert_eq!(pass.name, "raytrace_gi");
    assert_eq!(pass.pass_type, PassType::RayTracing);
    assert!(pass.color_attachments.is_empty(), "RT has no colour attachments");
    assert!(pass.depth_stencil.is_none(), "RT has no depth-stencil");
    assert!(pass.dispatch_source.is_some(), "RT has a dispatch source");
    assert_eq!(pass.view_type, ViewType::Storage);
}

#[test]
fn ir_pass_ray_tracing_indirect_dispatch() {
    let pass = IrPass::ray_tracing(
        PassIndex(5),
        "rt_indirect",
        DispatchSource::Indirect {
            buffer: ResourceHandle(30),
            offset: 128,
        },
    );

    match pass.dispatch_source {
        Some(DispatchSource::Indirect { buffer, offset }) => {
            assert_eq!(buffer, ResourceHandle(30));
            assert_eq!(offset, 128);
        }
        _ => panic!("Expected Indirect dispatch for RT"),
    }
}

// =============================================================================
// SECTION 18 -- IrPass: sync_access_set_from_attachments
// =============================================================================

#[test]
fn ir_pass_sync_access_set_populates_reads_and_writes_from_color_attachments() {
    let ca = ColorAttachment {
        resource: ResourceHandle(5),
        load_op: AttachmentLoadOp::Load,
        store_op: AttachmentStoreOp::Store,
        ..Default::default()
    };

    let pass = IrPass::graphics(
        PassIndex(0),
        "sync_test",
        vec![ca],
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

    // Load -> read, Store -> write
    assert!(
        pass.access_set.reads.contains(&ResourceHandle(5)),
        "Load op adds resource to reads"
    );
    assert!(
        pass.access_set.writes.contains(&ResourceHandle(5)),
        "Store op adds resource to writes"
    );
}

#[test]
fn ir_pass_sync_access_set_clear_load_does_not_add_read() {
    let ca = ColorAttachment {
        resource: ResourceHandle(1),
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        ..Default::default()
    };

    let pass = IrPass::graphics(
        PassIndex(0),
        "clear_only",
        vec![ca],
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

    assert!(
        !pass.access_set.reads.contains(&ResourceHandle(1)),
        "Clear load does NOT add to reads"
    );
    assert!(
        pass.access_set.writes.contains(&ResourceHandle(1)),
        "Store still adds to writes"
    );
}

#[test]
fn ir_pass_sync_access_set_dont_care_store_does_not_add_write() {
    let ca = ColorAttachment {
        resource: ResourceHandle(2),
        load_op: AttachmentLoadOp::Load,
        store_op: AttachmentStoreOp::DontCare,
        ..Default::default()
    };

    let pass = IrPass::graphics(
        PassIndex(0),
        "discard_write",
        vec![ca],
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

    assert!(
        pass.access_set.reads.contains(&ResourceHandle(2)),
        "Load still adds to reads"
    );
    assert!(
        !pass.access_set.writes.contains(&ResourceHandle(2)),
        "DontCare store does NOT add to writes"
    );
}

#[test]
fn ir_pass_sync_access_set_depth_stencil_adds_both_channels() {
    let ca = ColorAttachment {
        resource: ResourceHandle(1),
        ..Default::default()
    };
    let ds = DepthStencilAttachment {
        resource: ResourceHandle(3),
        depth_load_op: AttachmentLoadOp::Load,
        depth_store_op: AttachmentStoreOp::Store,
        stencil_load_op: AttachmentLoadOp::Load,
        stencil_store_op: AttachmentStoreOp::Store,
        ..Default::default()
    };

    let pass = IrPass::graphics(
        PassIndex(0),
        "depth_stencil_sync",
        vec![ca],
        Some(ds),
        InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    // Color attachment (Load+Store) -> reads + writes
    assert!(pass.access_set.reads.contains(&ResourceHandle(1)));
    assert!(pass.access_set.writes.contains(&ResourceHandle(1)));
    // Depth-stencil (dual Load+Store) -> reads + writes
    assert!(pass.access_set.reads.contains(&ResourceHandle(3)));
    assert!(pass.access_set.writes.contains(&ResourceHandle(3)));
}

#[test]
fn ir_pass_sync_access_set_depth_only_stencil_dont_care() {
    let ca = ColorAttachment {
        resource: ResourceHandle(1),
        ..Default::default()
    };
    let ds = DepthStencilAttachment {
        resource: ResourceHandle(2),
        depth_load_op: AttachmentLoadOp::Load,
        depth_store_op: AttachmentStoreOp::Store,
        stencil_load_op: AttachmentLoadOp::DontCare,
        stencil_store_op: AttachmentStoreOp::DontCare,
        ..Default::default()
    };

    let pass = IrPass::graphics(
        PassIndex(0),
        "depth_only",
        vec![ca],
        Some(ds),
        InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );

    // Depth Load+Store -> reads + writes
    assert!(pass.access_set.reads.contains(&ResourceHandle(2)));
    assert!(pass.access_set.writes.contains(&ResourceHandle(2)));
}

#[test]
fn ir_pass_sync_access_set_manual_resync_after_mutation() {
    let mut pass = IrPass::graphics(
        PassIndex(0),
        "mutate_test",
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

    // Initially empty.
    assert!(pass.access_set.is_empty(), "No attachments -> empty access set");

    // Mutate: add colour attachment manually.
    pass.color_attachments.push(ColorAttachment {
        resource: ResourceHandle(99),
        load_op: AttachmentLoadOp::Load,
        store_op: AttachmentStoreOp::Store,
        ..Default::default()
    });

    // Re-sync.
    pass.sync_access_set_from_attachments();

    assert!(
        pass.access_set.reads.contains(&ResourceHandle(99)),
        "After resync, reads contains newly added attachment"
    );
    assert!(
        pass.access_set.writes.contains(&ResourceHandle(99)),
        "After resync, writes contains newly added attachment"
    );
}

// =============================================================================
// SECTION 19 -- IrPass: has_color_attachments / has_depth_stencil / has_dispatch
// =============================================================================

#[test]
fn ir_pass_has_color_attachments_true_when_attachments_present() {
    let ca = ColorAttachment {
        resource: ResourceHandle(1),
        ..Default::default()
    };
    let pass = IrPass::graphics(
        PassIndex(0),
        "color_pass",
        vec![ca],
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
    assert!(pass.has_color_attachments(), "Graphics pass with colour attachments");
}

#[test]
fn ir_pass_has_color_attachments_false_when_empty() {
    let pass = IrPass::graphics(
        PassIndex(0),
        "empty_color",
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
    assert!(!pass.has_color_attachments(), "No colour attachments -> false");
}

#[test]
fn ir_pass_has_color_attachments_false_for_compute() {
    let pass = IrPass::compute(
        PassIndex(1),
        "compute_only",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    assert!(!pass.has_color_attachments(), "Compute has no colour attachments");
}

#[test]
fn ir_pass_has_color_attachments_false_for_copy() {
    let pass = IrPass::copy(PassIndex(2), "copy_only");
    assert!(!pass.has_color_attachments(), "Copy has no colour attachments");
}

#[test]
fn ir_pass_has_depth_stencil_true_when_some() {
    let ca = ColorAttachment {
        resource: ResourceHandle(1),
        ..Default::default()
    };
    let ds = DepthStencilAttachment {
        resource: ResourceHandle(2),
        ..Default::default()
    };
    let pass = IrPass::graphics(
        PassIndex(0),
        "depth_pass",
        vec![ca],
        Some(ds),
        InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        },
        ViewType::Texture2D,
    );
    assert!(pass.has_depth_stencil(), "Has depth-stencil");
}

#[test]
fn ir_pass_has_depth_stencil_false_when_none() {
    let ca = ColorAttachment {
        resource: ResourceHandle(1),
        ..Default::default()
    };
    let pass = IrPass::graphics(
        PassIndex(0),
        "no_depth",
        vec![ca],
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
    assert!(!pass.has_depth_stencil(), "No depth-stencil -> false");
}

#[test]
fn ir_pass_has_dispatch_true_for_compute() {
    let pass = IrPass::compute(
        PassIndex(1),
        "compute_dispatch",
        DispatchSource::Direct {
            group_count_x: 4,
            group_count_y: 4,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    assert!(pass.has_dispatch(), "Compute has dispatch source");
}

#[test]
fn ir_pass_has_dispatch_true_for_ray_tracing() {
    let pass = IrPass::ray_tracing(
        PassIndex(2),
        "rt_dispatch",
        DispatchSource::Direct {
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
    );
    assert!(pass.has_dispatch(), "Ray tracing has dispatch source");
}

#[test]
fn ir_pass_has_dispatch_false_for_graphics() {
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
        ViewType::Texture2D,
    );
    assert!(!pass.has_dispatch(), "Graphics has no dispatch source");
}

#[test]
fn ir_pass_has_dispatch_false_for_copy() {
    let pass = IrPass::copy(PassIndex(3), "copy_op");
    assert!(!pass.has_dispatch(), "Copy has no dispatch source");
}

// =============================================================================
// SECTION 20 -- IrPass: tags, Display
// =============================================================================

#[test]
fn ir_pass_tags_field_accessible_and_mutable() {
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

    assert!(pass.tags.is_empty(), "No tags initially");

    pass.tags.push("transparent".into());
    pass.tags.push("post-process".into());
    pass.tags.push("debug".into());

    assert_eq!(pass.tags.len(), 3, "Three tags added");
    assert!(pass.tags.contains(&"transparent".into()));
    assert!(pass.tags.contains(&"post-process".into()));
    assert!(pass.tags.contains(&"debug".into()));
}

#[test]
fn ir_pass_display_contains_key_information() {
    let pass = IrPass::graphics(
        PassIndex(0),
        "my_graphics_pass",
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

    let s = format!("{}", pass);
    assert!(s.contains("my_graphics_pass"), "Display contains pass name");
    assert!(s.contains("Graphics"), "Display contains pass type");
    assert!(s.contains("0"), "Display contains pass index");
}

// =============================================================================
// SECTION 21 -- EdgeType: variant discrimination, Display
// =============================================================================

#[test]
fn edge_type_all_variants() {
    assert_eq!(EdgeType::RAW, EdgeType::RAW);
    assert_eq!(EdgeType::WAR, EdgeType::WAR);
    assert_eq!(EdgeType::WAW, EdgeType::WAW);
    assert_ne!(EdgeType::RAW, EdgeType::WAR);
    assert_ne!(EdgeType::RAW, EdgeType::WAW);
    assert_ne!(EdgeType::WAR, EdgeType::WAW);
}

#[test]
fn edge_type_display() {
    assert_eq!(format!("{}", EdgeType::RAW), "RAW");
    assert_eq!(format!("{}", EdgeType::WAR), "WAR");
    assert_eq!(format!("{}", EdgeType::WAW), "WAW");
}

// =============================================================================
// SECTION 22 -- IrEdge: new(), all fields, Display
// =============================================================================

#[test]
fn ir_edge_new_constructs_with_all_fields() {
    let edge = IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(7), EdgeType::RAW);

    assert_eq!(edge.from, PassIndex(0), "From pass index");
    assert_eq!(edge.to, PassIndex(1), "To pass index");
    assert_eq!(edge.resource, ResourceHandle(7), "Resource handle");
    assert_eq!(edge.edge_type, EdgeType::RAW, "Edge type");
}

#[test]
fn ir_edge_war_type() {
    let edge = IrEdge::new(PassIndex(0), PassIndex(2), ResourceHandle(5), EdgeType::WAR);
    assert_eq!(edge.edge_type, EdgeType::WAR, "WAR edge type");
}

#[test]
fn ir_edge_waw_type() {
    let edge = IrEdge::new(PassIndex(1), PassIndex(3), ResourceHandle(9), EdgeType::WAW);
    assert_eq!(edge.edge_type, EdgeType::WAW, "WAW edge type");
}

#[test]
fn ir_edge_same_pass_different_resources() {
    let edge_a = IrEdge::new(PassIndex(0), PassIndex(2), ResourceHandle(1), EdgeType::RAW);
    let edge_b = IrEdge::new(PassIndex(0), PassIndex(2), ResourceHandle(2), EdgeType::RAW);

    assert_eq!(edge_a.from, edge_b.from, "Same from pass");
    assert_eq!(edge_a.to, edge_b.to, "Same to pass");
    assert_ne!(
        edge_a.resource, edge_b.resource,
        "Different resources produce different edges"
    );
}

#[test]
fn ir_edge_chain_accumulation() {
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), ResourceHandle(2), EdgeType::RAW),
        IrEdge::new(PassIndex(2), PassIndex(3), ResourceHandle(3), EdgeType::RAW),
    ];

    assert_eq!(edges.len(), 3, "Three edges in chain");
    for (i, edge) in edges.iter().enumerate() {
        assert_eq!(edge.from, PassIndex(i), "Edge {} from pass {}", i, i);
        assert_eq!(edge.to, PassIndex(i + 1), "Edge {} to pass {}", i, i + 1);
    }
}

#[test]
fn ir_edge_display_contains_all_fields() {
    let edge = IrEdge::new(PassIndex(2), PassIndex(5), ResourceHandle(3), EdgeType::WAW);
    let s = format!("{}", edge);

    assert!(s.contains("2"), "Display shows from index");
    assert!(s.contains("5"), "Display shows to index");
    assert!(s.contains("WAW"), "Display shows edge type");
    assert!(s.contains("3"), "Display shows resource handle");
}

// =============================================================================
// SECTION 23 -- Integration: pass-resource-edge round trip
// =============================================================================

#[test]
fn ir_round_trip_graphics_to_compute_with_raw_edge() {
    // Simulate a real frame graph segment:
    //   Pass 0 (Graphics) writes resource 1 as colour attachment
    //   Pass 1 (Compute) reads resource 1 as storage
    //   Edge: RAW from Pass 0 to Pass 1 over resource 1

    let res = IrResource::new(
        ResourceHandle(1),
        "shared_rt",
        ResourceDesc::Texture2D(TextureDesc {
            width: 256,
            height: 256,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba16float".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    let write_pass = IrPass::graphics(
        PassIndex(0),
        "write_pass",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 0.0],
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

    let mut read_pass = IrPass::compute(
        PassIndex(1),
        "read_pass",
        DispatchSource::Direct {
            group_count_x: 4,
            group_count_y: 4,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    read_pass.access_set.reads.push(ResourceHandle(1));

    let edge = IrEdge::new(write_pass.index, read_pass.index, res.handle, EdgeType::RAW);

    // Verify edge captures the dependency.
    assert_eq!(edge.from, PassIndex(0));
    assert_eq!(edge.to, PassIndex(1));
    assert_eq!(edge.resource, ResourceHandle(1));
    assert_eq!(edge.edge_type, EdgeType::RAW);

    // Verify write pass owns the resource in writes.
    assert!(write_pass.access_set.writes.contains(&ResourceHandle(1)));

    // Verify read pass owns the resource in reads.
    assert!(read_pass.access_set.reads.contains(&ResourceHandle(1)));

    // Verify resource metadata.
    assert_eq!(res.handle, ResourceHandle(1));
    assert_eq!(res.name, "shared_rt");
    assert_eq!(res.lifetime, ResourceLifetime::Transient);
}

#[test]
fn ir_round_trip_three_pass_chain_with_all_dependency_types() {
    // Build a 3-pass chain exercising RAW, WAR, and WAW edges:
    //   Pass 0 (Graphics) writes resource A (RAW producer)
    //   Pass 1 (Compute) reads resource A then writes resource B
    //   Pass 2 (Compute) reads resource B (RAW) and overwrites resource A (WAW)

    // Resources.
    let res_a = IrResource::new(
        ResourceHandle(1),
        "intermediate_a",
        ResourceDesc::Texture2D(TextureDesc {
            width: 128,
            height: 128,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba16float".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    let res_b = IrResource::new(
        ResourceHandle(2),
        "intermediate_b",
        ResourceDesc::Buffer(BufferDesc {
            size: 4096,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    // Pass 0: writes resource A
    let pass_0 = IrPass::graphics(
        PassIndex(0),
        "gbuffer",
        vec![ColorAttachment {
            resource: ResourceHandle(1),
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
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

    // Pass 1: reads A (RAW from pass 0), writes B
    let mut pass_1 = IrPass::compute(
        PassIndex(1),
        "lighting",
        DispatchSource::Direct {
            group_count_x: 8,
            group_count_y: 8,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass_1.access_set.reads.push(ResourceHandle(1));
    pass_1.access_set.writes.push(ResourceHandle(2));

    // Pass 2: reads B (RAW from pass 1), overwrites A (WAW from pass 0)
    let mut pass_2 = IrPass::compute(
        PassIndex(2),
        "post_process",
        DispatchSource::Direct {
            group_count_x: 16,
            group_count_y: 16,
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass_2.access_set.reads.push(ResourceHandle(2));
    pass_2.access_set.writes.push(ResourceHandle(1));

    // Edges.
    let edge_raw = IrEdge::new(PassIndex(0), PassIndex(1), ResourceHandle(1), EdgeType::RAW);
    let edge_raw_b = IrEdge::new(PassIndex(1), PassIndex(2), ResourceHandle(2), EdgeType::RAW);
    let edge_waw_a = IrEdge::new(PassIndex(0), PassIndex(2), ResourceHandle(1), EdgeType::WAW);

    // Verify all edges.
    assert_eq!(edge_raw.edge_type, EdgeType::RAW);
    assert_eq!(edge_raw_b.edge_type, EdgeType::RAW);
    assert_eq!(edge_waw_a.edge_type, EdgeType::WAW);

    // Verify access sets.
    assert!(pass_0.access_set.writes.contains(&ResourceHandle(1)));
    assert!(pass_1.access_set.reads.contains(&ResourceHandle(1)));
    assert!(pass_1.access_set.writes.contains(&ResourceHandle(2)));
    assert!(pass_2.access_set.reads.contains(&ResourceHandle(2)));
    assert!(pass_2.access_set.writes.contains(&ResourceHandle(1)));

    // Verify resource metadata.
    assert_eq!(res_a.handle, ResourceHandle(1));
    assert_eq!(res_a.name, "intermediate_a");
    assert_eq!(res_b.handle, ResourceHandle(2));
    assert_eq!(res_b.name, "intermediate_b");
}

#[test]
fn ir_round_trip_imported_swapchain() {
    // Simulate an imported swap chain resource with a single present pass.

    let swapchain = IrResource::new(
        ResourceHandle(0),
        "swapchain",
        ResourceDesc::Texture2D(TextureDesc {
            width: 1920,
            height: 1080,
            mip_levels: 1,
            array_layers: 1,
            format: "bgra8unorm-srgb".into(),
        }),
        ResourceLifetime::Imported,
        ResourceState::Present,
    );

    let present_pass = IrPass::graphics(
        PassIndex(0),
        "present",
        vec![ColorAttachment {
            resource: ResourceHandle(0),
            load_op: AttachmentLoadOp::Load,
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

    assert_eq!(swapchain.lifetime, ResourceLifetime::Imported);
    assert_eq!(swapchain.initial_state, ResourceState::Present);

    assert!(present_pass.access_set.reads.contains(&ResourceHandle(0)));
    assert!(present_pass.access_set.writes.contains(&ResourceHandle(0)));
}

#[test]
fn ir_round_trip_async_compute_chain() {
    // Simulate async compute: a compute pass reads from a graphics pass's output.

    let result = IrResource::new(
        ResourceHandle(1),
        "simulation_result",
        ResourceDesc::Buffer(BufferDesc {
            size: 262144,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    );

    let gfx = IrPass::graphics(
        PassIndex(0),
        "scene_render",
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
    // Graphics pass writes to the buffer via storage access.
    let mut gfx = gfx;
    gfx.access_set.writes.push(ResourceHandle(1));

    let mut async_comp = IrPass::compute(
        PassIndex(1),
        "async_sim",
        DispatchSource::Indirect {
            buffer: ResourceHandle(2),
            offset: 0,
        },
        ViewType::StorageBuffer,
    );
    async_comp.access_set.reads.push(ResourceHandle(1));

    let edge = IrEdge::new(gfx.index, async_comp.index, result.handle, EdgeType::RAW);

    assert_eq!(edge.edge_type, EdgeType::RAW);
    assert_eq!(async_comp.view_type, ViewType::StorageBuffer);
    match async_comp.dispatch_source {
        Some(DispatchSource::Indirect { buffer, offset }) => {
            assert_eq!(buffer, ResourceHandle(2));
            assert_eq!(offset, 0);
        }
        _ => panic!("Expected Indirect dispatch"),
    }
}
