// Blackbox contract tests for T-FG-4.5 wgpu barrier command generation.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// The module under test (`wgpu_barriers`) provides bitflag types mirroring
// wgpu's TextureUsages / BufferUsages, a ResourceState → wgpu-usage mapping
// function, a resolved barrier enum, a batch generation function, and a
// resolve context.
//
// Coverage:
//   1.  WgpuTextureUsage constants are accessible and distinct
//   2.  WgpuBufferUsage constants are accessible and distinct
//   3.  resource_state_to_wgpu_usage(ShaderRead) returns TEXTURE_BINDING | UNIFORM
//   4.  resource_state_to_wgpu_usage(ColorAttachment) returns RENDER_ATTACHMENT
//   5.  resource_state_to_wgpu_usage(DepthStencilAttachment) returns RENDER_ATTACHMENT
//   6.  All 13 ResourceState variants have mappings (no panics)
//   7.  WgpuBarrier::Texture can be constructed
//   8.  WgpuBarrier::Buffer can be constructed
//   9.  generate_wgpu_barriers() with valid input produces barriers
//  10.  Empty input produces empty vec
//  11.  WgpuBarrierResolveContext can be constructed

use renderer_backend::frame_graph::{
    WgpuBarrier, WgpuTextureUsage, WgpuBufferUsage,
    resource_state_to_wgpu_usage, generate_wgpu_barriers,
    WgpuBarrierResolveContext,
    ResourceState, ResourceHandle, PassIndex, EdgeType,
    IrResource, BarrierDescriptor,
    TextureBarrierDescriptor, BufferBarrierDescriptor, BarrierCommand,
    mock_resource_texture, mock_resource_buffer,
};

// =========================================================================
// SECTION 1 -- WgpuTextureUsage constants
// =========================================================================

#[test]
fn wgpu_texture_usage_empty_returns_zero() {
    assert_eq!(WgpuTextureUsage::empty().bits(), 0);
}

#[test]
fn wgpu_texture_usage_constants_have_expected_bit_values() {
    assert_eq!(WgpuTextureUsage::COPY_SRC.bits(), 1);
    assert_eq!(WgpuTextureUsage::COPY_DST.bits(), 2);
    assert_eq!(WgpuTextureUsage::TEXTURE_BINDING.bits(), 4);
    assert_eq!(WgpuTextureUsage::STORAGE_BINDING.bits(), 8);
    assert_eq!(WgpuTextureUsage::RENDER_ATTACHMENT.bits(), 16);
    assert_eq!(WgpuTextureUsage::PRESENT.bits(), 32);
}

#[test]
fn wgpu_texture_usage_constants_are_distinct() {
    let mut seen = std::collections::HashSet::new();
    for bits in &[
        WgpuTextureUsage::COPY_SRC.bits(),
        WgpuTextureUsage::COPY_DST.bits(),
        WgpuTextureUsage::TEXTURE_BINDING.bits(),
        WgpuTextureUsage::STORAGE_BINDING.bits(),
        WgpuTextureUsage::RENDER_ATTACHMENT.bits(),
        WgpuTextureUsage::PRESENT.bits(),
    ] {
        assert!(seen.insert(bits), "duplicate bit value {}", bits);
    }
}

#[test]
fn wgpu_texture_usage_contains_works() {
    let combined = WgpuTextureUsage::COPY_SRC | WgpuTextureUsage::COPY_DST;
    assert!(combined.contains(WgpuTextureUsage::COPY_SRC));
    assert!(combined.contains(WgpuTextureUsage::COPY_DST));
    assert!(!combined.contains(WgpuTextureUsage::TEXTURE_BINDING));
}

#[test]
fn wgpu_texture_usage_insert_combines_flags() {
    let mut usage = WgpuTextureUsage::empty();
    usage.insert(WgpuTextureUsage::RENDER_ATTACHMENT);
    assert!(usage.contains(WgpuTextureUsage::RENDER_ATTACHMENT));
    assert!(!usage.contains(WgpuTextureUsage::COPY_SRC));

    usage.insert(WgpuTextureUsage::COPY_SRC);
    assert!(usage.contains(WgpuTextureUsage::RENDER_ATTACHMENT));
    assert!(usage.contains(WgpuTextureUsage::COPY_SRC));
}

// =========================================================================
// SECTION 2 -- WgpuBufferUsage constants
// =========================================================================

#[test]
fn wgpu_buffer_usage_empty_returns_zero() {
    assert_eq!(WgpuBufferUsage::empty().bits(), 0);
}

#[test]
fn wgpu_buffer_usage_constants_have_expected_bit_values() {
    assert_eq!(WgpuBufferUsage::COPY_SRC.bits(), 1);
    assert_eq!(WgpuBufferUsage::COPY_DST.bits(), 2);
    assert_eq!(WgpuBufferUsage::INDEX.bits(), 4);
    assert_eq!(WgpuBufferUsage::VERTEX.bits(), 8);
    assert_eq!(WgpuBufferUsage::UNIFORM.bits(), 16);
    assert_eq!(WgpuBufferUsage::STORAGE.bits(), 32);
    assert_eq!(WgpuBufferUsage::INDIRECT.bits(), 64);
}

#[test]
fn wgpu_buffer_usage_constants_are_distinct() {
    let mut seen = std::collections::HashSet::new();
    for bits in &[
        WgpuBufferUsage::COPY_SRC.bits(),
        WgpuBufferUsage::COPY_DST.bits(),
        WgpuBufferUsage::INDEX.bits(),
        WgpuBufferUsage::VERTEX.bits(),
        WgpuBufferUsage::UNIFORM.bits(),
        WgpuBufferUsage::STORAGE.bits(),
        WgpuBufferUsage::INDIRECT.bits(),
    ] {
        assert!(seen.insert(bits), "duplicate bit value {}", bits);
    }
}

#[test]
fn wgpu_buffer_usage_contains_works() {
    let combined = WgpuBufferUsage::VERTEX | WgpuBufferUsage::INDEX;
    assert!(combined.contains(WgpuBufferUsage::VERTEX));
    assert!(combined.contains(WgpuBufferUsage::INDEX));
    assert!(!combined.contains(WgpuBufferUsage::UNIFORM));
}

#[test]
fn wgpu_buffer_usage_insert_combines_flags() {
    let mut usage = WgpuBufferUsage::empty();
    usage.insert(WgpuBufferUsage::STORAGE);
    assert!(usage.contains(WgpuBufferUsage::STORAGE));

    usage.insert(WgpuBufferUsage::COPY_DST);
    assert!(usage.contains(WgpuBufferUsage::STORAGE));
    assert!(usage.contains(WgpuBufferUsage::COPY_DST));
}

// =========================================================================
// SECTION 3 -- resource_state_to_wgpu_usage mapping
// =========================================================================

#[test]
fn shader_read_maps_to_texture_binding_and_uniform() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::ShaderRead);
    assert!(tex.contains(WgpuTextureUsage::TEXTURE_BINDING));
    assert!(!tex.contains(WgpuTextureUsage::STORAGE_BINDING));
    assert!(buf.contains(WgpuBufferUsage::UNIFORM));
    assert!(!buf.contains(WgpuBufferUsage::STORAGE));
}

#[test]
fn color_attachment_maps_to_render_attachment() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::ColorAttachment);
    assert!(tex.contains(WgpuTextureUsage::RENDER_ATTACHMENT));
    assert!(!tex.contains(WgpuTextureUsage::TEXTURE_BINDING));
    assert_eq!(buf.bits(), 0, "buffer usage should be empty");
}

#[test]
fn depth_stencil_attachment_maps_to_render_attachment() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::DepthStencilAttachment);
    assert!(tex.contains(WgpuTextureUsage::RENDER_ATTACHMENT));
    assert!(!tex.contains(WgpuTextureUsage::PRESENT));
    assert_eq!(buf.bits(), 0, "buffer usage should be empty");
}

#[test]
fn uninitialized_maps_to_empty() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::Uninitialized);
    assert_eq!(tex.bits(), 0);
    assert_eq!(buf.bits(), 0);
}

#[test]
fn vertex_buffer_maps_to_vertex() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::VertexBuffer);
    assert_eq!(tex.bits(), 0, "texture usage should be empty for VertexBuffer");
    assert!(buf.contains(WgpuBufferUsage::VERTEX));
    assert!(!buf.contains(WgpuBufferUsage::INDEX));
}

#[test]
fn index_buffer_maps_to_index() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::IndexBuffer);
    assert_eq!(tex.bits(), 0, "texture usage should be empty for IndexBuffer");
    assert!(buf.contains(WgpuBufferUsage::INDEX));
    assert!(!buf.contains(WgpuBufferUsage::VERTEX));
}

#[test]
fn indirect_argument_maps_to_indirect() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::IndirectArgument);
    assert_eq!(tex.bits(), 0, "texture usage should be empty for IndirectArgument");
    assert!(buf.contains(WgpuBufferUsage::INDIRECT));
    assert!(!buf.contains(WgpuBufferUsage::STORAGE));
}

#[test]
fn depth_stencil_read_only_maps_to_texture_binding() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::DepthStencilReadOnly);
    assert!(tex.contains(WgpuTextureUsage::TEXTURE_BINDING));
    assert!(!tex.contains(WgpuTextureUsage::RENDER_ATTACHMENT));
    assert_eq!(buf.bits(), 0, "buffer usage should be empty");
}

#[test]
fn shader_read_write_maps_to_storage_binding_and_storage() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::ShaderReadWrite);
    assert!(tex.contains(WgpuTextureUsage::STORAGE_BINDING));
    assert!(buf.contains(WgpuBufferUsage::STORAGE));
}

#[test]
fn transfer_src_maps_to_copy_src() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::TransferSrc);
    assert!(tex.contains(WgpuTextureUsage::COPY_SRC));
    assert!(buf.contains(WgpuBufferUsage::COPY_SRC));
}

#[test]
fn transfer_dst_maps_to_copy_dst() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::TransferDst);
    assert!(tex.contains(WgpuTextureUsage::COPY_DST));
    assert!(buf.contains(WgpuBufferUsage::COPY_DST));
}

#[test]
fn acceleration_structure_maps_to_storage_binding_and_storage() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::AccelerationStructure);
    assert!(tex.contains(WgpuTextureUsage::STORAGE_BINDING));
    assert!(buf.contains(WgpuBufferUsage::STORAGE));
}

#[test]
fn present_maps_to_present() {
    let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::Present);
    assert!(tex.contains(WgpuTextureUsage::PRESENT));
    assert!(!tex.contains(WgpuTextureUsage::RENDER_ATTACHMENT));
    assert_eq!(buf.bits(), 0, "buffer usage should be empty");
}

/// All 13 ResourceState variants map without panicking.
#[test]
fn all_resource_state_variants_have_mappings() {
    let states = [
        ResourceState::Uninitialized,
        ResourceState::VertexBuffer,
        ResourceState::IndexBuffer,
        ResourceState::IndirectArgument,
        ResourceState::ColorAttachment,
        ResourceState::DepthStencilAttachment,
        ResourceState::DepthStencilReadOnly,
        ResourceState::ShaderRead,
        ResourceState::ShaderReadWrite,
        ResourceState::TransferSrc,
        ResourceState::TransferDst,
        ResourceState::AccelerationStructure,
        ResourceState::Present,
    ];

    for state in &states {
        let (tex, buf) = resource_state_to_wgpu_usage(*state);
        // No panic during mapping -- texture and buffer usages are valid.
        let _ = tex.bits();
        let _ = buf.bits();
    }
}

// =========================================================================
// SECTION 4 -- WgpuBarrier construction
// =========================================================================

#[test]
fn wgpu_barrier_texture_can_be_constructed() {
    let barrier = WgpuBarrier::Texture {
        resource: ResourceHandle(1),
        from: WgpuTextureUsage::empty(),
        to: WgpuTextureUsage::RENDER_ATTACHMENT,
        mip_levels: None,
        array_layers: None,
    };
    assert_eq!(barrier.resource(), ResourceHandle(1));
}

#[test]
fn wgpu_barrier_texture_with_subresource_ranges() {
    let barrier = WgpuBarrier::Texture {
        resource: ResourceHandle(5),
        from: WgpuTextureUsage::RENDER_ATTACHMENT,
        to: WgpuTextureUsage::TEXTURE_BINDING,
        mip_levels: Some(0..4),
        array_layers: Some(0..1),
    };
    assert_eq!(barrier.resource(), ResourceHandle(5));
}

#[test]
fn wgpu_barrier_buffer_can_be_constructed() {
    let barrier = WgpuBarrier::Buffer {
        resource: ResourceHandle(10),
        from: WgpuBufferUsage::empty(),
        to: WgpuBufferUsage::UNIFORM,
        offset: None,
        size: None,
    };
    assert_eq!(barrier.resource(), ResourceHandle(10));
}

#[test]
fn wgpu_barrier_buffer_with_byte_range() {
    let barrier = WgpuBarrier::Buffer {
        resource: ResourceHandle(20),
        from: WgpuBufferUsage::STORAGE,
        to: WgpuBufferUsage::VERTEX,
        offset: Some(0),
        size: Some(4096),
    };
    assert_eq!(barrier.resource(), ResourceHandle(20));
}

#[test]
fn wgpu_barrier_debug_format() {
    let barrier = WgpuBarrier::Texture {
        resource: ResourceHandle(1),
        from: WgpuTextureUsage::empty(),
        to: WgpuTextureUsage::RENDER_ATTACHMENT,
        mip_levels: None,
        array_layers: None,
    };
    let s = format!("{:?}", barrier);
    assert!(s.contains("Texture"));
    assert!(s.contains("ResourceHandle(1)"));
}

#[test]
fn wgpu_barrier_clone_and_partial_eq() {
    let a = WgpuBarrier::Texture {
        resource: ResourceHandle(3),
        from: WgpuTextureUsage::empty(),
        to: WgpuTextureUsage::TEXTURE_BINDING,
        mip_levels: None,
        array_layers: None,
    };
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn wgpu_barrier_texture_vs_buffer_not_equal() {
    let tex = WgpuBarrier::Texture {
        resource: ResourceHandle(1),
        from: WgpuTextureUsage::empty(),
        to: WgpuTextureUsage::RENDER_ATTACHMENT,
        mip_levels: None,
        array_layers: None,
    };
    let buf = WgpuBarrier::Buffer {
        resource: ResourceHandle(1),
        from: WgpuBufferUsage::empty(),
        to: WgpuBufferUsage::UNIFORM,
        offset: None,
        size: None,
    };
    assert_ne!(tex, buf);
}

// =========================================================================
// SECTION 5 -- generate_wgpu_barriers
// =========================================================================

#[test]
fn generate_wgpu_barriers_empty_input_returns_empty_vec() {
    let result = generate_wgpu_barriers(&[], &[]);
    assert!(result.is_empty());
}

#[test]
fn generate_wgpu_barriers_with_valid_texture_produces_barrier() {
    let handle = ResourceHandle(1);
    let resources = vec![mock_resource_texture(handle, "color", 1920, 1080)];
    let barriers = vec![(
        PassIndex(0),
        PassIndex(1),
        handle,
        EdgeType::RAW,
        ResourceState::Uninitialized,
        ResourceState::ColorAttachment,
    )];

    let result = generate_wgpu_barriers(&barriers, &resources);

    assert_eq!(result.len(), 1, "one barrier tuple should produce one WgpuBarrier");

    match &result[0] {
        WgpuBarrier::Texture { resource, from, to, .. } => {
            assert_eq!(*resource, handle);
            assert_eq!(from.bits(), 0, "Uninitialized maps to empty texture usage");
            assert!(to.contains(WgpuTextureUsage::RENDER_ATTACHMENT));
        }
        other => panic!("expected WgpuBarrier::Texture, got {:?}", other),
    }
}

#[test]
fn generate_wgpu_barriers_with_valid_buffer_produces_barrier() {
    let handle = ResourceHandle(10);
    let resources = vec![mock_resource_buffer(handle, "storage_buf", 4096)];
    let barriers = vec![(
        PassIndex(0),
        PassIndex(1),
        handle,
        EdgeType::RAW,
        ResourceState::Uninitialized,
        ResourceState::VertexBuffer,
    )];

    let result = generate_wgpu_barriers(&barriers, &resources);

    assert_eq!(result.len(), 1, "one buffer barrier tuple should produce one WgpuBarrier");

    match &result[0] {
        WgpuBarrier::Buffer { resource, from, to, .. } => {
            assert_eq!(*resource, handle);
            assert_eq!(from.bits(), 0, "Uninitialized maps to empty buffer usage");
            assert!(to.contains(WgpuBufferUsage::VERTEX));
        }
        other => panic!("expected WgpuBarrier::Buffer, got {:?}", other),
    }
}

#[test]
fn generate_wgpu_barriers_skips_unknown_handle() {
    let handle = ResourceHandle(1);
    let resources = vec![mock_resource_texture(handle, "color", 64, 64)];
    let barriers = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceHandle(999), // not in resources
        EdgeType::RAW,
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    )];

    let result = generate_wgpu_barriers(&barriers, &resources);
    assert!(result.is_empty(), "unknown handle should be skipped");
}

#[test]
fn generate_wgpu_barriers_multiple_barriers() {
    let tex_handle = ResourceHandle(1);
    let buf_handle = ResourceHandle(10);
    let resources = vec![
        mock_resource_texture(tex_handle, "color", 1920, 1080),
        mock_resource_buffer(buf_handle, "storage", 8192),
    ];
    let barriers = vec![
        (
            PassIndex(0),
            PassIndex(1),
            tex_handle,
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ColorAttachment,
        ),
        (
            PassIndex(0),
            PassIndex(1),
            buf_handle,
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ShaderRead,
        ),
    ];

    let result = generate_wgpu_barriers(&barriers, &resources);
    assert_eq!(result.len(), 2, "two barrier tuples should produce two WgpuBarriers");

    // First should be a texture barrier (tex_handle), second a buffer barrier.
    assert!(
        matches!(&result[0], WgpuBarrier::Texture { .. }),
        "first barrier should be texture"
    );
    assert!(
        matches!(&result[1], WgpuBarrier::Buffer { .. }),
        "second barrier should be buffer"
    );
}

// =========================================================================
// SECTION 6 -- WgpuBarrierResolveContext
// =========================================================================

#[test]
fn wgpu_barrier_resolve_context_new_with_empty_slice() {
    let ctx = WgpuBarrierResolveContext::new(&[]);
    let r = ctx.is_texture(ResourceHandle(1));
    assert_eq!(r, None, "empty context returns None for any handle");
}

#[test]
fn wgpu_barrier_resolve_context_new_with_texture_resource() {
    let resources = vec![mock_resource_texture(ResourceHandle(1), "tex", 800, 600)];
    let ctx = WgpuBarrierResolveContext::new(&resources);

    let is_tex = ctx.is_texture(ResourceHandle(1));
    assert_eq!(is_tex, Some(true), "texture resource should report as texture");
}

#[test]
fn wgpu_barrier_resolve_context_new_with_buffer_resource() {
    let resources = vec![mock_resource_buffer(ResourceHandle(10), "buf", 2048)];
    let ctx = WgpuBarrierResolveContext::new(&resources);

    let is_tex = ctx.is_texture(ResourceHandle(10));
    assert_eq!(is_tex, Some(false), "buffer resource should NOT report as texture");
}

#[test]
fn wgpu_barrier_resolve_context_unknown_handle_returns_none() {
    let resources = vec![mock_resource_texture(ResourceHandle(1), "tex", 800, 600)];
    let ctx = WgpuBarrierResolveContext::new(&resources);

    let r = ctx.is_texture(ResourceHandle(999));
    assert_eq!(r, None, "unknown handle returns None");
}

#[test]
fn wgpu_barrier_resolve_context_resolve_texture_descriptor() {
    let resources = vec![mock_resource_texture(ResourceHandle(1), "color", 1920, 1080)];
    let ctx = WgpuBarrierResolveContext::new(&resources);

    let desc = TextureBarrierDescriptor {
        resource: ResourceHandle(1),
        before: ResourceState::Uninitialized,
        after: ResourceState::ShaderRead,
        mip_levels: None,
        array_layers: None,
    };
    let barrier = ctx.resolve(&BarrierDescriptor::Texture(desc));

    assert!(barrier.is_some(), "known texture should resolve");
    match barrier.unwrap() {
        WgpuBarrier::Texture { resource, .. } => {
            assert_eq!(resource, ResourceHandle(1));
        }
        other => panic!("expected Texture barrier, got {:?}", other),
    }
}

#[test]
fn wgpu_barrier_resolve_context_resolve_buffer_descriptor() {
    let resources = vec![mock_resource_buffer(ResourceHandle(10), "data", 4096)];
    let ctx = WgpuBarrierResolveContext::new(&resources);

    let desc = BufferBarrierDescriptor {
        resource: ResourceHandle(10),
        before: ResourceState::Uninitialized,
        after: ResourceState::ShaderRead,
        offset: None,
        size: None,
    };
    let barrier = ctx.resolve(&BarrierDescriptor::Buffer(desc));

    assert!(barrier.is_some(), "known buffer should resolve");
    match barrier.unwrap() {
        WgpuBarrier::Buffer { resource, .. } => {
            assert_eq!(resource, ResourceHandle(10));
        }
        other => panic!("expected Buffer barrier, got {:?}", other),
    }
}

#[test]
fn wgpu_barrier_resolve_context_resolve_unknown_returns_none() {
    let resources = vec![mock_resource_texture(ResourceHandle(1), "tex", 64, 64)];
    let ctx = WgpuBarrierResolveContext::new(&resources);

    let desc = TextureBarrierDescriptor {
        resource: ResourceHandle(999),
        before: ResourceState::Uninitialized,
        after: ResourceState::ShaderRead,
        mip_levels: None,
        array_layers: None,
    };
    let result = ctx.resolve(&BarrierDescriptor::Texture(desc));
    assert!(result.is_none(), "unknown handle should return None");
}

#[test]
fn wgpu_barrier_resolve_context_resolve_batch() {
    let resources = vec![
        mock_resource_texture(ResourceHandle(1), "color", 1920, 1080),
        mock_resource_buffer(ResourceHandle(10), "data", 4096),
    ];
    let ctx = WgpuBarrierResolveContext::new(&resources);

    let cmd = BarrierCommand {
        texture_barriers: vec![TextureBarrierDescriptor {
            resource: ResourceHandle(1),
            before: ResourceState::Uninitialized,
            after: ResourceState::ShaderRead,
            mip_levels: None,
            array_layers: None,
        }],
        buffer_barriers: vec![BufferBarrierDescriptor {
            resource: ResourceHandle(10),
            before: ResourceState::Uninitialized,
            after: ResourceState::ShaderRead,
            offset: None,
            size: None,
        }],
    };

    let barriers = ctx.resolve_batch(&cmd);
    assert_eq!(barriers.len(), 2, "batch with 1 texture + 1 buffer = 2 barriers");

    // Order: texture first, then buffer (from the resolve_batch implementation).
    assert!(matches!(&barriers[0], WgpuBarrier::Texture { .. }));
    assert!(matches!(&barriers[1], WgpuBarrier::Buffer { .. }));
}
