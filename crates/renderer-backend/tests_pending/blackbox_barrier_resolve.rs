// Blackbox contract tests for T-FG-4.6 BarrierResolveContext.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// DISABLED: BarrierResolveContext is not yet implemented in the public API.
// These tests will be re-enabled once T-FG-4.6 is complete.
//
// Acceptance criteria (contract):
//   BarrierResolveContext::new(&[IrResource]) builds a descriptor map.
//   resolve_texture_barrier(handle, before, after) returns:
//     - Some(TextureBarrierDescriptor) for a known texture resource
//     - None for an unknown handle
//     - None for a buffer resource (wrong type)
//   resolve_buffer_barrier(handle, before, after) returns:
//     - Some(BufferBarrierDescriptor) for a known buffer resource
//     - None for an unknown handle
//     - None for a texture resource (wrong type)
//
// Coverage:
//   1.  Construction -- new() accepts &[IrResource]
//   2.  Debug + Clone traits
//   3.  Texture resolve -- known Texture2D returns Some
//   4.  Texture resolve -- Texture3D returns Some
//   5.  Texture resolve -- TextureCube returns Some
//   6.  Texture resolve -- unknown handle returns None
//   7.  Texture resolve -- buffer resource returns None
//   8.  Buffer resolve  -- known buffer returns Some
//   9.  Buffer resolve  -- unknown handle returns None
//  10.  Buffer resolve  -- texture resource returns None
//  11.  Mixed resources -- texture and buffer in same map, each resolves correctly
//  12.  Empty resources -- always returns None
//  13.  NONE handle     -- ResourceHandle::NONE returns None
//  14.  Fields preserved -- returned texture descriptor matches input fields
//  15.  Fields preserved -- returned buffer descriptor matches input fields
//  16.  State transitions -- various before/after pairs produce correct descriptors
//  17.  Same-state transition -- before == after still produces valid descriptor
//  18.  Multiple textures -- each resolves independently

// TODO(T-FG-4.6): Re-enable when BarrierResolveContext is implemented
#![allow(dead_code)]

// =========================================================================
// ALL TESTS DISABLED: BarrierResolveContext not yet implemented
// =========================================================================

/*
use renderer_backend::frame_graph::{
    BarrierResolveContext, BufferDesc, IrResource, ResourceDesc,
    ResourceHandle, ResourceLifetime, ResourceState, TextureDesc,
    Texture3DDesc,
};

// =========================================================================
// SECTION 1 -- Construction, Debug, Clone
// =========================================================================

#[test]
fn barrier_resolve_context_new_accepts_empty_slice() {
    let ctx = BarrierResolveContext::new(&[]);
    // No resources -- every resolve returns None; the test is that new() doesn't panic.
    assert_eq!(
        ctx.resolve_texture_barrier(ResourceHandle(1), ResourceState::Uninitialized, ResourceState::ShaderRead),
        None,
    );
}

#[test]
fn barrier_resolve_context_new_with_resources() {
    let resources = vec![
        make_texture(ResourceHandle(1), "color", 1920, 1080),
        make_buffer(ResourceHandle(10), "storage", 4096),
    ];
    let ctx = BarrierResolveContext::new(&resources);
    let tex = ctx.resolve_texture_barrier(
        ResourceHandle(1),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    assert!(tex.is_some(), "texture resolve from non-empty context");
}

#[test]
fn barrier_resolve_context_debug_format() {
    let ctx = BarrierResolveContext::new(&[]);
    let s = format!("{:?}", ctx);
    assert!(s.contains("BarrierResolveContext"), "Debug output contains type name");
}

#[test]
fn barrier_resolve_context_clone_produces_independent_instance() {
    let resources = vec![
        make_texture(ResourceHandle(1), "tex", 800, 600),
        make_buffer(ResourceHandle(10), "buf", 2048),
    ];
    let a = BarrierResolveContext::new(&resources);
    let b = a.clone();

    // Both should behave identically.
    let tex_a = a.resolve_texture_barrier(
        ResourceHandle(1),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    let tex_b = b.resolve_texture_barrier(
        ResourceHandle(1),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    assert_eq!(tex_a, tex_b, "clone preserves texture resolution");

    let buf_a = a.resolve_buffer_barrier(
        ResourceHandle(10),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    let buf_b = b.resolve_buffer_barrier(
        ResourceHandle(10),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    assert_eq!(buf_a, buf_b, "clone preserves buffer resolution");
}

// =========================================================================
// SECTION 2 -- resolve_texture_barrier: known textures
// =========================================================================

#[test]
fn resolve_texture_known_handle_returns_some() {
    let resources = vec![make_texture(ResourceHandle(1), "color", 1920, 1080)];
    let ctx = BarrierResolveContext::new(&resources);
    let result = ctx.resolve_texture_barrier(
        ResourceHandle(1),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    assert!(result.is_some(), "known texture handle should resolve");
}

#[test]
fn resolve_texture_texture2d_returns_texture_barrier() {
    let resources = vec![make_texture(ResourceHandle(1), "color", 1920, 1080)];
    let ctx = BarrierResolveContext::new(&resources);
    let result = ctx.resolve_texture_barrier(
        ResourceHandle(1),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    let desc = result.expect("Texture2D should resolve");
    assert_eq!(desc.resource, ResourceHandle(1));
    assert_eq!(desc.before, ResourceState::Uninitialized);
    assert_eq!(desc.after, ResourceState::ShaderRead);
    assert!(desc.mip_levels.is_none(), "full subresource by default");
    assert!(desc.array_layers.is_none(), "full subresource by default");
}

#[test]
fn resolve_texture_texture3d_returns_texture_barrier() {
    let resources = vec![IrResource::new(
        ResourceHandle(2),
        "volume",
        ResourceDesc::Texture3D(Texture3DDesc {
            width: 256,
            height: 256,
            depth: 64,
            mip_levels: 1,
            format: "r32float".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )];
    let ctx = BarrierResolveContext::new(&resources);
    let result = ctx.resolve_texture_barrier(
        ResourceHandle(2),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    let desc = result.expect("Texture3D should resolve");
    assert_eq!(desc.resource, ResourceHandle(2));
    assert_eq!(desc.before, ResourceState::Uninitialized);
    assert_eq!(desc.after, ResourceState::ShaderRead);
}

#[test]
fn resolve_texture_texture_cube_returns_texture_barrier() {
    let resources = vec![IrResource::new(
        ResourceHandle(3),
        "cubemap",
        ResourceDesc::TextureCube(TextureDesc {
            width: 512,
            height: 512,
            mip_levels: 1,
            array_layers: 6,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )];
    let ctx = BarrierResolveContext::new(&resources);
    let result = ctx.resolve_texture_barrier(
        ResourceHandle(3),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    let desc = result.expect("TextureCube should resolve");
    assert_eq!(desc.resource, ResourceHandle(3));
}

// =========================================================================
// SECTION 3 -- resolve_texture_barrier: unknown / wrong type
// =========================================================================

#[test]
fn resolve_texture_unknown_handle_returns_none() {
    let resources = vec![make_texture(ResourceHandle(1), "color", 800, 600)];
    let ctx = BarrierResolveContext::new(&resources);
    let result = ctx.resolve_texture_barrier(
        ResourceHandle(99),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    assert!(result.is_none(), "unknown handle returns None");
}

#[test]
fn resolve_texture_buffer_resource_returns_none() {
    let resources = vec![make_buffer(ResourceHandle(10), "storage", 4096)];
    let ctx = BarrierResolveContext::new(&resources);
    let result = ctx.resolve_texture_barrier(
        ResourceHandle(10),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    assert!(result.is_none(), "buffer resource queried as texture returns None");
}

#[test]
fn resolve_texture_mixed_resources_only_textures_resolve() {
    let resources = vec![
        make_texture(ResourceHandle(1), "color", 800, 600),
        make_buffer(ResourceHandle(10), "storage", 4096),
        make_texture(ResourceHandle(2), "depth", 800, 600),
        make_buffer(ResourceHandle(20), "vbuf", 8192),
    ];
    let ctx = BarrierResolveContext::new(&resources);

    // Textures should resolve.
    assert!(ctx.resolve_texture_barrier(
        ResourceHandle(1),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ).is_some(), "Texture2D resolves as texture");

    assert!(ctx.resolve_texture_barrier(
        ResourceHandle(2),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ).is_some(), "second texture resolves as texture");

    // Buffers should NOT resolve as textures.
    assert!(ctx.resolve_texture_barrier(
        ResourceHandle(10),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ).is_none(), "buffer does not resolve as texture");

    assert!(ctx.resolve_texture_barrier(
        ResourceHandle(20),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ).is_none(), "second buffer does not resolve as texture");
}

// =========================================================================
// SECTION 4 -- resolve_buffer_barrier: known buffers
// =========================================================================

#[test]
fn resolve_buffer_known_handle_returns_some() {
    let resources = vec![make_buffer(ResourceHandle(10), "storage", 4096)];
    let ctx = BarrierResolveContext::new(&resources);
    let result = ctx.resolve_buffer_barrier(
        ResourceHandle(10),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    assert!(result.is_some(), "known buffer handle should resolve");
}

#[test]
fn resolve_buffer_returns_buffer_barrier() {
    let resources = vec![make_buffer(ResourceHandle(10), "storage", 4096)];
    let ctx = BarrierResolveContext::new(&resources);
    let result = ctx.resolve_buffer_barrier(
        ResourceHandle(10),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    let desc = result.expect("buffer should resolve");
    assert_eq!(desc.resource, ResourceHandle(10));
    assert_eq!(desc.before, ResourceState::Uninitialized);
    assert_eq!(desc.after, ResourceState::ShaderRead);
    assert!(desc.offset.is_none(), "full subresource by default");
    assert!(desc.size.is_none(), "full subresource by default");
}

#[test]
fn resolve_buffer_indirect_arg_buffer_resolves() {
    let resources = vec![IrResource::new(
        ResourceHandle(15),
        "indirect",
        ResourceDesc::Buffer(BufferDesc {
            size: 256,
            usage: "indirect".into(),
            is_indirect_arg: true,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )];
    let ctx = BarrierResolveContext::new(&resources);
    let result = ctx.resolve_buffer_barrier(
        ResourceHandle(15),
        ResourceState::Uninitialized,
        ResourceState::IndirectArgument,
    );
    let desc = result.expect("indirect arg buffer should resolve");
    assert_eq!(desc.resource, ResourceHandle(15));
    assert_eq!(desc.after, ResourceState::IndirectArgument);
}

// =========================================================================
// SECTION 5 -- resolve_buffer_barrier: unknown / wrong type
// =========================================================================

#[test]
fn resolve_buffer_unknown_handle_returns_none() {
    let resources = vec![make_buffer(ResourceHandle(10), "storage", 4096)];
    let ctx = BarrierResolveContext::new(&resources);
    let result = ctx.resolve_buffer_barrier(
        ResourceHandle(99),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    assert!(result.is_none(), "unknown handle returns None");
}

#[test]
fn resolve_buffer_texture_resource_returns_none() {
    let resources = vec![make_texture(ResourceHandle(1), "color", 800, 600)];
    let ctx = BarrierResolveContext::new(&resources);
    let result = ctx.resolve_buffer_barrier(
        ResourceHandle(1),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    assert!(result.is_none(), "texture resource queried as buffer returns None");
}

#[test]
fn resolve_buffer_mixed_resources_only_buffers_resolve() {
    let resources = vec![
        make_buffer(ResourceHandle(10), "storage", 4096),
        make_texture(ResourceHandle(1), "color", 800, 600),
        make_buffer(ResourceHandle(20), "vbuf", 8192),
    ];
    let ctx = BarrierResolveContext::new(&resources);

    // Buffers should resolve.
    assert!(ctx.resolve_buffer_barrier(
        ResourceHandle(10),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ).is_some(), "buffer resolves as buffer");

    assert!(ctx.resolve_buffer_barrier(
        ResourceHandle(20),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ).is_some(), "second buffer resolves as buffer");

    // Textures should NOT resolve as buffers.
    assert!(ctx.resolve_buffer_barrier(
        ResourceHandle(1),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ).is_none(), "texture does not resolve as buffer");
}

// =========================================================================
// SECTION 6 -- Edge cases: empty, NONE, no resources
// =========================================================================

#[test]
fn resolve_empty_resources_always_none() {
    let ctx = BarrierResolveContext::new(&[]);
    assert!(ctx.resolve_texture_barrier(
        ResourceHandle(1),
        ResourceState::ShaderRead,
        ResourceState::ColorAttachment,
    ).is_none(), "empty: texture resolve returns None");
    assert!(ctx.resolve_buffer_barrier(
        ResourceHandle(1),
        ResourceState::ShaderRead,
        ResourceState::ColorAttachment,
    ).is_none(), "empty: buffer resolve returns None");
}

#[test]
fn resolve_none_handle_returns_none() {
    let resources = vec![make_texture(ResourceHandle(1), "tex", 800, 600)];
    let ctx = BarrierResolveContext::new(&resources);
    assert!(ctx.resolve_texture_barrier(
        ResourceHandle::NONE,
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ).is_none(), "ResourceHandle::NONE returns None for texture");
    assert!(ctx.resolve_buffer_barrier(
        ResourceHandle::NONE,
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    ).is_none(), "ResourceHandle::NONE returns None for buffer");
}

// =========================================================================
// SECTION 7 -- State transitions on texture resolve
// =========================================================================

#[test]
fn resolve_texture_preserves_all_state_fields() {
    let resources = vec![make_texture(ResourceHandle(5), "albedo", 1920, 1080)];
    let ctx = BarrierResolveContext::new(&resources);

    let transitions = vec![
        (ResourceState::Uninitialized, ResourceState::ShaderRead),
        (ResourceState::ShaderRead, ResourceState::ColorAttachment),
        (ResourceState::ColorAttachment, ResourceState::ShaderRead),
        (ResourceState::ShaderRead, ResourceState::TransferSrc),
        (ResourceState::TransferDst, ResourceState::ShaderRead),
        (ResourceState::DepthStencilAttachment, ResourceState::ShaderRead),
        (ResourceState::ShaderRead, ResourceState::DepthStencilAttachment),
        (ResourceState::ShaderReadWrite, ResourceState::ShaderRead),
        (ResourceState::ShaderRead, ResourceState::Present),
    ];

    for (before, after) in &transitions {
        let result = ctx.resolve_texture_barrier(ResourceHandle(5), *before, *after);
        let desc = result.unwrap_or_else(|| {
            panic!("resolve_texture_barrier({:?} -> {:?}) should be Some", before, after)
        });
        assert_eq!(desc.resource, ResourceHandle(5), "handle preserved");
        assert_eq!(desc.before, *before, "before state preserved for {:?} -> {:?}", before, after);
        assert_eq!(desc.after, *after, "after state preserved for {:?} -> {:?}", before, after);
    }
}

#[test]
fn resolve_texture_same_state_still_produces_barrier() {
    // resolve_texture_barrier is a resolve, NOT an optimizer -- same-state
    // transitions should still produce a TextureBarrierDescriptor.
    let resources = vec![make_texture(ResourceHandle(1), "tex", 800, 600)];
    let ctx = BarrierResolveContext::new(&resources);

    let same_states = vec![
        ResourceState::Uninitialized,
        ResourceState::ColorAttachment,
        ResourceState::DepthStencilAttachment,
        ResourceState::DepthStencilReadOnly,
        ResourceState::ShaderRead,
        ResourceState::ShaderReadWrite,
        ResourceState::TransferSrc,
        ResourceState::TransferDst,
        ResourceState::Present,
    ];

    for state in &same_states {
        let result = ctx.resolve_texture_barrier(ResourceHandle(1), *state, *state);
        assert!(
            result.is_some(),
            "same-state {:?} -> {:?} still produces a barrier descriptor",
            state,
            state,
        );
    }
}

// =========================================================================
// SECTION 8 -- State transitions on buffer resolve
// =========================================================================

#[test]
fn resolve_buffer_preserves_all_state_fields() {
    let resources = vec![make_buffer(ResourceHandle(10), "ssbo", 8192)];
    let ctx = BarrierResolveContext::new(&resources);

    let transitions = vec![
        (ResourceState::Uninitialized, ResourceState::VertexBuffer),
        (ResourceState::VertexBuffer, ResourceState::ShaderRead),
        (ResourceState::ShaderRead, ResourceState::ShaderReadWrite),
        (ResourceState::ShaderReadWrite, ResourceState::TransferSrc),
        (ResourceState::TransferDst, ResourceState::ShaderRead),
        (ResourceState::ShaderRead, ResourceState::IndexBuffer),
        (ResourceState::IndexBuffer, ResourceState::IndirectArgument),
        (ResourceState::Uninitialized, ResourceState::TransferDst),
    ];

    for (before, after) in &transitions {
        let result = ctx.resolve_buffer_barrier(ResourceHandle(10), *before, *after);
        let desc = result.unwrap_or_else(|| {
            panic!("resolve_buffer_barrier({:?} -> {:?}) should be Some", before, after)
        });
        assert_eq!(desc.resource, ResourceHandle(10), "handle preserved");
        assert_eq!(desc.before, *before, "before state preserved");
        assert_eq!(desc.after, *after, "after state preserved");
    }
}

#[test]
fn resolve_buffer_same_state_still_produces_barrier() {
    let resources = vec![make_buffer(ResourceHandle(10), "buf", 4096)];
    let ctx = BarrierResolveContext::new(&resources);

    let same_states = vec![
        ResourceState::Uninitialized,
        ResourceState::VertexBuffer,
        ResourceState::IndexBuffer,
        ResourceState::IndirectArgument,
        ResourceState::ShaderRead,
        ResourceState::ShaderReadWrite,
        ResourceState::TransferSrc,
        ResourceState::TransferDst,
    ];

    for state in &same_states {
        let result = ctx.resolve_buffer_barrier(ResourceHandle(10), *state, *state);
        assert!(
            result.is_some(),
            "same-state {:?} -> {:?} still produces a buffer barrier descriptor",
            state,
            state,
        );
    }
}

// =========================================================================
// SECTION 9 -- Independent resolution across multiple textures
// =========================================================================

#[test]
fn resolve_texture_multiple_textures_resolve_independently() {
    let resources = vec![
        make_texture(ResourceHandle(1), "albedo", 1920, 1080),
        make_texture(ResourceHandle(2), "normal", 1920, 1080),
        make_texture(ResourceHandle(3), "depth", 1920, 1080),
    ];
    let ctx = BarrierResolveContext::new(&resources);

    let r1 = ctx.resolve_texture_barrier(
        ResourceHandle(1),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    let r2 = ctx.resolve_texture_barrier(
        ResourceHandle(2),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );
    let r3 = ctx.resolve_texture_barrier(
        ResourceHandle(3),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
    );

    assert!(r1.is_some(), "texture 1 resolves");
    assert!(r2.is_some(), "texture 2 resolves");
    assert!(r3.is_some(), "texture 3 resolves");

    assert_eq!(r1.unwrap().resource, ResourceHandle(1));
    assert_eq!(r2.unwrap().resource, ResourceHandle(2));
    assert_eq!(r3.unwrap().resource, ResourceHandle(3));
}

#[test]
fn resolve_buffer_multiple_buffers_resolve_independently() {
    let resources = vec![
        make_buffer(ResourceHandle(10), "ssbo_a", 4096),
        make_buffer(ResourceHandle(20), "ssbo_b", 8192),
        make_buffer(ResourceHandle(30), "vbuf", 65536),
    ];
    let ctx = BarrierResolveContext::new(&resources);

    let r1 = ctx.resolve_buffer_barrier(
        ResourceHandle(10),
        ResourceState::Uninitialized,
        ResourceState::ShaderReadWrite,
    );
    let r2 = ctx.resolve_buffer_barrier(
        ResourceHandle(20),
        ResourceState::Uninitialized,
        ResourceState::VertexBuffer,
    );
    let r3 = ctx.resolve_buffer_barrier(
        ResourceHandle(30),
        ResourceState::Uninitialized,
        ResourceState::IndexBuffer,
    );

    assert!(r1.is_some(), "buffer 10 resolves");
    assert!(r2.is_some(), "buffer 20 resolves");
    assert!(r3.is_some(), "buffer 30 resolves");

    assert_eq!(r1.unwrap().resource, ResourceHandle(10));
    assert_eq!(r2.unwrap().resource, ResourceHandle(20));
    assert_eq!(r3.unwrap().resource, ResourceHandle(30));
}

// =========================================================================
// SECTION 10 -- Cross-type rejection: texture as buffer, buffer as texture
// =========================================================================

#[test]
fn resolve_cross_type_rejection_all_variants() {
    // Every texture variant should be rejected by resolve_buffer_barrier.
    let resources = vec![
        IrResource::new(
            ResourceHandle(1),
            "tex2d",
            ResourceDesc::Texture2D(TextureDesc {
                width: 800, height: 600, mip_levels: 1, array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(2),
            "tex3d",
            ResourceDesc::Texture3D(Texture3DDesc {
                width: 256, height: 256, depth: 64, mip_levels: 1,
                format: "r32float".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        IrResource::new(
            ResourceHandle(3),
            "cube",
            ResourceDesc::TextureCube(TextureDesc {
                width: 512, height: 512, mip_levels: 1, array_layers: 6,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ),
        make_buffer(ResourceHandle(10), "buf", 4096),
    ];
    let ctx = BarrierResolveContext::new(&resources);

    // resolve_buffer_barrier on all three texture types returns None.
    assert!(ctx.resolve_buffer_barrier(ResourceHandle(1), ResourceState::Uninitialized, ResourceState::ShaderRead).is_none(),
        "Texture2D rejected by resolve_buffer_barrier");
    assert!(ctx.resolve_buffer_barrier(ResourceHandle(2), ResourceState::Uninitialized, ResourceState::ShaderRead).is_none(),
        "Texture3D rejected by resolve_buffer_barrier");
    assert!(ctx.resolve_buffer_barrier(ResourceHandle(3), ResourceState::Uninitialized, ResourceState::ShaderRead).is_none(),
        "TextureCube rejected by resolve_buffer_barrier");

    // resolve_texture_barrier on buffer returns None.
    assert!(ctx.resolve_texture_barrier(ResourceHandle(10), ResourceState::Uninitialized, ResourceState::ShaderRead).is_none(),
        "Buffer rejected by resolve_texture_barrier");
}

// =========================================================================
// Helpers
// =========================================================================

fn make_texture(handle: ResourceHandle, name: &str, width: u32, height: u32) -> IrResource {
    IrResource::new(
        handle,
        name,
        ResourceDesc::Texture2D(TextureDesc {
            width,
            height,
            mip_levels: 1,
            array_layers: 1,
            format: "rgba8unorm".into(),
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}

fn make_buffer(handle: ResourceHandle, name: &str, size: u64) -> IrResource {
    IrResource::new(
        handle,
        name,
        ResourceDesc::Buffer(BufferDesc {
            size,
            usage: "storage".into(),
            is_indirect_arg: false,
        }),
        ResourceLifetime::Transient,
        ResourceState::Uninitialized,
    )
}
*/ // End of disabled test block
