// Blackbox contract tests for T-FG-4.5 Barrier descriptor generation.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria:
//   - Barrier descriptors can be created for common GPU state transitions
//     (ShaderRead → ColorAttachment, ColorAttachment → ShaderRead, etc.)
//   - BarrierDescriptor correctly discriminates texture vs buffer barriers
//   - BarrierCommand bundles texture + buffer barriers for a pass boundary
//   - wgpu_barrier_from_state_transition dispatches to the right variant
//   - generate_barriers() produces correct output for known pass graphs
//   - All ResourceState variants produce valid barrier descriptors
//   - Empty input produces empty output
//
// Coverage:
//   1.  TextureBarrierDescriptor -- full field construction, Debug, Clone, PartialEq
//   2.  BufferBarrierDescriptor  -- full field construction, Debug, Clone, PartialEq
//   3.  BarrierDescriptor        -- Texture/Buffer variants, resource() accessor
//   4.  BarrierCommand           -- Default, field bundling, collect, Debug/Clone/PartialEq
//   5.  wgpu_barrier_from_state_transition -- texture dispatch, buffer dispatch
//   6.  resource_state_to_texture_usage    -- every valid state + panic on invalid
//   7.  resource_state_to_buffer_usage     -- every valid state + panic on invalid
//   8.  generate_barriers -- empty input, single boundary, mixed barrier types
//   9.  Full API chain    -- compute_barriers → generate_barriers → BarrierCommand

use renderer_backend::frame_graph::{
    BarrierCommand, BarrierDescriptor, BufferBarrierDescriptor, TextureBarrierDescriptor,
    ResourceState, ResourceHandle, ResourceDesc, IrEdge,
    PassIndex, EdgeType, TextureDesc, BufferDesc, compute_barriers, generate_barriers,
    wgpu_barrier_from_state_transition, resource_state_to_texture_usage,
    resource_state_to_buffer_usage, mock_pass_graphics, mock_pass_compute,
    mock_resource_texture, mock_resource_buffer,
};

// ===== SECTION 1 -- TextureBarrierDescriptor =====

#[test]
fn texture_barrier_descriptor_constructs_with_all_fields() {
    let desc = TextureBarrierDescriptor {
        resource: ResourceHandle(1),
        before: ResourceState::ShaderRead,
        after: ResourceState::ColorAttachment,
        mip_levels: Some(0..4),
        array_layers: Some(0..1),
    };
    assert_eq!(desc.resource, ResourceHandle(1));
    assert_eq!(desc.before, ResourceState::ShaderRead);
    assert_eq!(desc.after, ResourceState::ColorAttachment);
    assert_eq!(desc.mip_levels, Some(0..4));
    assert_eq!(desc.array_layers, Some(0..1));
}

#[test]
fn texture_barrier_descriptor_mip_levels_none_means_full_resource() {
    let desc = TextureBarrierDescriptor {
        resource: ResourceHandle(2),
        before: ResourceState::ColorAttachment,
        after: ResourceState::ShaderRead,
        mip_levels: None,
        array_layers: None,
    };
    assert!(desc.mip_levels.is_none());
    assert!(desc.array_layers.is_none());
}

#[test]
fn texture_barrier_descriptor_debug_format() {
    let desc = TextureBarrierDescriptor {
        resource: ResourceHandle(1),
        before: ResourceState::Uninitialized,
        after: ResourceState::TransferDst,
        mip_levels: None,
        array_layers: None,
    };
    let s = format!("{:?}", desc);
    assert!(s.contains("TextureBarrierDescriptor"));
    assert!(s.contains("Uninitialized"));
    assert!(s.contains("TransferDst"));
}

#[test]
fn texture_barrier_descriptor_clone_equality() {
    let a = TextureBarrierDescriptor {
        resource: ResourceHandle(5),
        before: ResourceState::ColorAttachment,
        after: ResourceState::ShaderRead,
        mip_levels: None,
        array_layers: None,
    };
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn texture_barrier_descriptor_partial_eq_distinguishes_resource() {
    let base = TextureBarrierDescriptor {
        resource: ResourceHandle(1),
        before: ResourceState::ShaderRead,
        after: ResourceState::ColorAttachment,
        mip_levels: None,
        array_layers: None,
    };
    let different = TextureBarrierDescriptor {
        resource: ResourceHandle(2),
        ..base.clone()
    };
    assert_ne!(base, different);
}

#[test]
fn texture_barrier_descriptor_partial_eq_distinguishes_state() {
    let base = TextureBarrierDescriptor {
        resource: ResourceHandle(1),
        before: ResourceState::ShaderRead,
        after: ResourceState::ColorAttachment,
        mip_levels: None,
        array_layers: None,
    };
    let different = TextureBarrierDescriptor {
        after: ResourceState::ShaderRead,
        ..base.clone()
    };
    assert_ne!(base, different);
}

#[test]
fn texture_barrier_descriptor_partial_eq_distinguishes_subresource() {
    let base = TextureBarrierDescriptor {
        resource: ResourceHandle(1),
        before: ResourceState::Uninitialized,
        after: ResourceState::TransferDst,
        mip_levels: Some(0..1),
        array_layers: None,
    };
    let different = TextureBarrierDescriptor {
        mip_levels: None,
        ..base.clone()
    };
    assert_ne!(base, different);
}

// ===== SECTION 2 -- BufferBarrierDescriptor =====

#[test]
fn buffer_barrier_descriptor_constructs_with_all_fields() {
    let desc = BufferBarrierDescriptor {
        resource: ResourceHandle(10),
        before: ResourceState::VertexBuffer,
        after: ResourceState::ShaderRead,
        offset: Some(0),
        size: Some(4096),
    };
    assert_eq!(desc.resource, ResourceHandle(10));
    assert_eq!(desc.before, ResourceState::VertexBuffer);
    assert_eq!(desc.after, ResourceState::ShaderRead);
    assert_eq!(desc.offset, Some(0));
    assert_eq!(desc.size, Some(4096));
}

#[test]
fn buffer_barrier_descriptor_offset_size_none_means_entire_buffer() {
    let desc = BufferBarrierDescriptor {
        resource: ResourceHandle(3),
        before: ResourceState::TransferDst,
        after: ResourceState::ShaderRead,
        offset: None,
        size: None,
    };
    assert!(desc.offset.is_none());
    assert!(desc.size.is_none());
}

#[test]
fn buffer_barrier_descriptor_debug_format() {
    let desc = BufferBarrierDescriptor {
        resource: ResourceHandle(2),
        before: ResourceState::IndexBuffer,
        after: ResourceState::IndirectArgument,
        offset: None,
        size: None,
    };
    let s = format!("{:?}", desc);
    assert!(s.contains("BufferBarrierDescriptor"));
    assert!(s.contains("IndexBuffer"));
    assert!(s.contains("IndirectArgument"));
}

#[test]
fn buffer_barrier_descriptor_clone_equality() {
    let a = BufferBarrierDescriptor {
        resource: ResourceHandle(7),
        before: ResourceState::TransferSrc,
        after: ResourceState::ShaderRead,
        offset: Some(128),
        size: Some(256),
    };
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn buffer_barrier_descriptor_partial_eq_distinguishes_offset() {
    let base = BufferBarrierDescriptor {
        resource: ResourceHandle(1),
        before: ResourceState::ShaderRead,
        after: ResourceState::TransferSrc,
        offset: None,
        size: None,
    };
    let with_offset = BufferBarrierDescriptor {
        offset: Some(64),
        ..base.clone()
    };
    assert_ne!(base, with_offset);
}

// ===== SECTION 3 -- BarrierDescriptor =====

#[test]
fn barrier_descriptor_texture_variant_holds_texture_descriptor() {
    let inner = TextureBarrierDescriptor {
        resource: ResourceHandle(1),
        before: ResourceState::ShaderRead,
        after: ResourceState::ColorAttachment,
        mip_levels: None,
        array_layers: None,
    };
    let desc = BarrierDescriptor::Texture(inner.clone());
    match &desc {
        BarrierDescriptor::Texture(t) => assert_eq!(*t, inner),
        BarrierDescriptor::Buffer(_) => panic!("expected Texture variant"),
    }
}

#[test]
fn barrier_descriptor_buffer_variant_holds_buffer_descriptor() {
    let inner = BufferBarrierDescriptor {
        resource: ResourceHandle(5),
        before: ResourceState::VertexBuffer,
        after: ResourceState::ShaderRead,
        offset: None,
        size: None,
    };
    let desc = BarrierDescriptor::Buffer(inner.clone());
    match &desc {
        BarrierDescriptor::Buffer(b) => assert_eq!(*b, inner),
        BarrierDescriptor::Texture(_) => panic!("expected Buffer variant"),
    }
}

#[test]
fn barrier_descriptor_resource_accessor_returns_handle() {
    let texture = BarrierDescriptor::Texture(TextureBarrierDescriptor {
        resource: ResourceHandle(42),
        before: ResourceState::Uninitialized,
        after: ResourceState::ShaderRead,
        mip_levels: None,
        array_layers: None,
    });
    assert_eq!(texture.resource(), ResourceHandle(42));

    let buffer = BarrierDescriptor::Buffer(BufferBarrierDescriptor {
        resource: ResourceHandle(99),
        before: ResourceState::TransferDst,
        after: ResourceState::ShaderRead,
        offset: None,
        size: None,
    });
    assert_eq!(buffer.resource(), ResourceHandle(99));
}

#[test]
fn barrier_descriptor_debug_format() {
    let desc = BarrierDescriptor::Texture(TextureBarrierDescriptor {
        resource: ResourceHandle(1),
        before: ResourceState::ShaderRead,
        after: ResourceState::ColorAttachment,
        mip_levels: None,
        array_layers: None,
    });
    let s = format!("{:?}", desc);
    assert!(s.contains("BarrierDescriptor"));
}

#[test]
fn barrier_descriptor_clone_equality() {
    let a = BarrierDescriptor::Texture(TextureBarrierDescriptor {
        resource: ResourceHandle(3),
        before: ResourceState::ColorAttachment,
        after: ResourceState::ShaderRead,
        mip_levels: None,
        array_layers: None,
    });
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn barrier_descriptor_texture_not_equal_buffer() {
    let tex = BarrierDescriptor::Texture(TextureBarrierDescriptor {
        resource: ResourceHandle(1),
        before: ResourceState::Uninitialized,
        after: ResourceState::ShaderRead,
        mip_levels: None,
        array_layers: None,
    });
    let buf = BarrierDescriptor::Buffer(BufferBarrierDescriptor {
        resource: ResourceHandle(1),
        before: ResourceState::Uninitialized,
        after: ResourceState::ShaderRead,
        offset: None,
        size: None,
    });
    assert_ne!(tex, buf);
}

// ===== SECTION 4 -- BarrierCommand =====

#[test]
fn barrier_command_default_is_empty() {
    let cmd = BarrierCommand::default();
    assert!(cmd.texture_barriers.is_empty());
    assert!(cmd.buffer_barriers.is_empty());
}

#[test]
fn barrier_command_holds_texture_barriers() {
    let cmd = BarrierCommand {
        texture_barriers: vec![
            TextureBarrierDescriptor {
                resource: ResourceHandle(1),
                before: ResourceState::ShaderRead,
                after: ResourceState::ColorAttachment,
                mip_levels: None,
                array_layers: None,
            },
        ],
        buffer_barriers: vec![],
    };
    assert_eq!(cmd.texture_barriers.len(), 1);
    assert!(cmd.buffer_barriers.is_empty());
}

#[test]
fn barrier_command_holds_buffer_barriers() {
    let cmd = BarrierCommand {
        texture_barriers: vec![],
        buffer_barriers: vec![
            BufferBarrierDescriptor {
                resource: ResourceHandle(10),
                before: ResourceState::VertexBuffer,
                after: ResourceState::ShaderRead,
                offset: None,
                size: None,
            },
        ],
    };
    assert!(cmd.texture_barriers.is_empty());
    assert_eq!(cmd.buffer_barriers.len(), 1);
}

#[test]
fn barrier_command_holds_mixed_barriers() {
    let cmd = BarrierCommand {
        texture_barriers: vec![
            TextureBarrierDescriptor {
                resource: ResourceHandle(1),
                before: ResourceState::ShaderRead,
                after: ResourceState::ColorAttachment,
                mip_levels: None,
                array_layers: None,
            },
        ],
        buffer_barriers: vec![
            BufferBarrierDescriptor {
                resource: ResourceHandle(10),
                before: ResourceState::TransferDst,
                after: ResourceState::ShaderRead,
                offset: None,
                size: None,
            },
        ],
    };
    assert_eq!(cmd.texture_barriers.len(), 1);
    assert_eq!(cmd.buffer_barriers.len(), 1);
}

#[test]
fn barrier_command_debug_format() {
    let cmd = BarrierCommand {
        texture_barriers: vec![],
        buffer_barriers: vec![],
    };
    let s = format!("{:?}", cmd);
    assert!(s.contains("BarrierCommand"));
}

#[test]
fn barrier_command_clone_equality() {
    let a = BarrierCommand {
        texture_barriers: vec![TextureBarrierDescriptor {
            resource: ResourceHandle(1),
            before: ResourceState::ShaderRead,
            after: ResourceState::ColorAttachment,
            mip_levels: None,
            array_layers: None,
        }],
        buffer_barriers: vec![],
    };
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn barrier_command_default_is_partial_eq_to_empty() {
    let empty = BarrierCommand {
        texture_barriers: vec![],
        buffer_barriers: vec![],
    };
    assert_eq!(BarrierCommand::default(), empty);
}

// ===== SECTION 5 -- wgpu_barrier_from_state_transition =====

#[test]
fn wgpu_barrier_from_texture_resource_returns_texture_descriptor() {
    let desc = ResourceDesc::Texture2D(TextureDesc {
        width: 1920,
        height: 1080,
        mip_levels: 1,
        array_layers: 1,
        format: "rgba8unorm".into(),
    });
    let barrier = wgpu_barrier_from_state_transition(
        ResourceHandle(1),
        ResourceState::ShaderRead,
        ResourceState::ColorAttachment,
        &desc,
    );
    match barrier {
        BarrierDescriptor::Texture(t) => {
            assert_eq!(t.resource, ResourceHandle(1));
            assert_eq!(t.before, ResourceState::ShaderRead);
            assert_eq!(t.after, ResourceState::ColorAttachment);
            assert!(t.mip_levels.is_none());
            assert!(t.array_layers.is_none());
        }
        BarrierDescriptor::Buffer(_) => panic!("expected Texture barrier for Texture2D resource"),
    }
}

#[test]
fn wgpu_barrier_from_texture3d_resource_returns_texture_descriptor() {
    let desc = ResourceDesc::Texture3D(
        renderer_backend::frame_graph::Texture3DDesc {
            width: 256,
            height: 256,
            depth: 64,
            mip_levels: 1,
            format: "r32float".into(),
        },
    );
    let barrier = wgpu_barrier_from_state_transition(
        ResourceHandle(2),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
        &desc,
    );
    assert!(matches!(barrier, BarrierDescriptor::Texture(_)));
}

#[test]
fn wgpu_barrier_from_texture_cube_resource_returns_texture_descriptor() {
    let desc = ResourceDesc::TextureCube(TextureDesc {
        width: 512,
        height: 512,
        mip_levels: 1,
        array_layers: 6,
        format: "rgba8unorm".into(),
    });
    let barrier = wgpu_barrier_from_state_transition(
        ResourceHandle(3),
        ResourceState::ShaderRead,
        ResourceState::ColorAttachment,
        &desc,
    );
    assert!(matches!(barrier, BarrierDescriptor::Texture(_)));
}

#[test]
fn wgpu_barrier_from_buffer_resource_returns_buffer_descriptor() {
    let desc = ResourceDesc::Buffer(BufferDesc {
        size: 4096,
        usage: "storage".into(),
        is_indirect_arg: false,
    });
    let barrier = wgpu_barrier_from_state_transition(
        ResourceHandle(10),
        ResourceState::VertexBuffer,
        ResourceState::ShaderRead,
        &desc,
    );
    match barrier {
        BarrierDescriptor::Buffer(b) => {
            assert_eq!(b.resource, ResourceHandle(10));
            assert_eq!(b.before, ResourceState::VertexBuffer);
            assert_eq!(b.after, ResourceState::ShaderRead);
            assert!(b.offset.is_none());
            assert!(b.size.is_none());
        }
        BarrierDescriptor::Texture(_) => panic!("expected Buffer barrier for Buffer resource"),
    }
}

#[test]
fn wgpu_barrier_from_transition_preserves_state_pair() {
    let desc = ResourceDesc::Texture2D(TextureDesc {
        width: 1, height: 1, mip_levels: 1, array_layers: 1, format: "r8unorm".into(),
    });
    let barrier = wgpu_barrier_from_state_transition(
        ResourceHandle(1),
        ResourceState::ColorAttachment,
        ResourceState::Present,
        &desc,
    );
    match barrier {
        BarrierDescriptor::Texture(t) => {
            assert_eq!(t.before, ResourceState::ColorAttachment);
            assert_eq!(t.after, ResourceState::Present);
        }
        _ => panic!("expected Texture variant"),
    }
}

#[test]
fn wgpu_barrier_from_transition_covers_common_gpu_transitions() {
    let tex_desc = ResourceDesc::Texture2D(TextureDesc {
        width: 1, height: 1, mip_levels: 1, array_layers: 1, format: "rgba8unorm".into(),
    });
    let buf_desc = ResourceDesc::Buffer(BufferDesc {
        size: 64, usage: "storage".into(), is_indirect_arg: false,
    });

    let transitions: Vec<(ResourceState, ResourceState, &ResourceDesc, &str)> = vec![
        (ResourceState::ShaderRead, ResourceState::ColorAttachment, &tex_desc, "texture: shader read -> color attachment"),
        (ResourceState::ColorAttachment, ResourceState::ShaderRead, &tex_desc, "texture: color attachment -> shader read"),
        (ResourceState::ShaderReadWrite, ResourceState::ShaderReadWrite, &tex_desc, "texture: UAV -> UAV (same state)"),
        (ResourceState::Uninitialized, ResourceState::TransferDst, &tex_desc, "texture: uninit -> copy dst"),
        (ResourceState::TransferDst, ResourceState::ShaderRead, &tex_desc, "texture: copy dst -> shader read"),
        (ResourceState::DepthStencilAttachment, ResourceState::ShaderRead, &tex_desc, "texture: depth -> shader read"),
        (ResourceState::ShaderRead, ResourceState::DepthStencilAttachment, &tex_desc, "texture: shader read -> depth"),
        (ResourceState::VertexBuffer, ResourceState::ShaderRead, &buf_desc, "buffer: vertex -> shader read"),
        (ResourceState::IndexBuffer, ResourceState::IndirectArgument, &buf_desc, "buffer: index -> indirect"),
        (ResourceState::TransferDst, ResourceState::ShaderReadWrite, &buf_desc, "buffer: copy dst -> storage"),
        (ResourceState::ShaderRead, ResourceState::TransferSrc, &buf_desc, "buffer: shader read -> copy src"),
    ];

    for (before, after, desc, label) in &transitions {
        let barrier = wgpu_barrier_from_state_transition(
            ResourceHandle(1),
            *before,
            *after,
            desc,
        );
        let is_texture = matches!(desc, ResourceDesc::Texture2D(_));
        match &barrier {
            BarrierDescriptor::Texture(t) => {
                assert!(is_texture, "{}: expected Texture, got Texture", label);
                assert_eq!(t.before, *before, "{}", label);
                assert_eq!(t.after, *after, "{}", label);
            }
            BarrierDescriptor::Buffer(b) => {
                assert!(!is_texture, "{}: expected Buffer, got Buffer", label);
                assert_eq!(b.before, *before, "{}", label);
                assert_eq!(b.after, *after, "{}", label);
            }
        }
    }
}

// ===== SECTION 6 -- resource_state_to_texture_usage =====

#[test]
fn texture_usage_color_attachment() {
    assert_eq!(
        resource_state_to_texture_usage(ResourceState::ColorAttachment),
        "RenderAttachment",
    );
}

#[test]
fn texture_usage_depth_stencil_attachment() {
    assert_eq!(
        resource_state_to_texture_usage(ResourceState::DepthStencilAttachment),
        "RenderAttachment",
    );
}

#[test]
fn texture_usage_depth_stencil_read_only() {
    assert_eq!(
        resource_state_to_texture_usage(ResourceState::DepthStencilReadOnly),
        "TextureBinding",
    );
}

#[test]
fn texture_usage_shader_read() {
    assert_eq!(
        resource_state_to_texture_usage(ResourceState::ShaderRead),
        "TextureBinding",
    );
}

#[test]
fn texture_usage_shader_read_write() {
    assert_eq!(
        resource_state_to_texture_usage(ResourceState::ShaderReadWrite),
        "StorageBinding",
    );
}

#[test]
fn texture_usage_transfer_src() {
    assert_eq!(
        resource_state_to_texture_usage(ResourceState::TransferSrc),
        "CopySrc",
    );
}

#[test]
fn texture_usage_transfer_dst() {
    assert_eq!(
        resource_state_to_texture_usage(ResourceState::TransferDst),
        "CopyDst",
    );
}

#[test]
fn texture_usage_present() {
    assert_eq!(
        resource_state_to_texture_usage(ResourceState::Present),
        "Present",
    );
}

#[test]
fn texture_usage_uninitialized() {
    assert_eq!(
        resource_state_to_texture_usage(ResourceState::Uninitialized),
        "(empty)",
    );
}

#[test]
#[should_panic(expected = "has no texture counterpart")]
fn texture_usage_panics_on_vertex_buffer() {
    resource_state_to_texture_usage(ResourceState::VertexBuffer);
}

#[test]
#[should_panic(expected = "has no texture counterpart")]
fn texture_usage_panics_on_index_buffer() {
    resource_state_to_texture_usage(ResourceState::IndexBuffer);
}

#[test]
#[should_panic(expected = "has no texture counterpart")]
fn texture_usage_panics_on_indirect_argument() {
    resource_state_to_texture_usage(ResourceState::IndirectArgument);
}

#[test]
#[should_panic(expected = "has no texture counterpart")]
fn texture_usage_panics_on_acceleration_structure() {
    resource_state_to_texture_usage(ResourceState::AccelerationStructure);
}

// ===== SECTION 7 -- resource_state_to_buffer_usage =====

#[test]
fn buffer_usage_vertex() {
    assert_eq!(
        resource_state_to_buffer_usage(ResourceState::VertexBuffer),
        "Vertex",
    );
}

#[test]
fn buffer_usage_index() {
    assert_eq!(
        resource_state_to_buffer_usage(ResourceState::IndexBuffer),
        "Index",
    );
}

#[test]
fn buffer_usage_indirect() {
    assert_eq!(
        resource_state_to_buffer_usage(ResourceState::IndirectArgument),
        "Indirect",
    );
}

#[test]
fn buffer_usage_shader_read() {
    assert_eq!(
        resource_state_to_buffer_usage(ResourceState::ShaderRead),
        "Uniform | TextureBinding",
    );
}

#[test]
fn buffer_usage_shader_read_write() {
    assert_eq!(
        resource_state_to_buffer_usage(ResourceState::ShaderReadWrite),
        "Storage",
    );
}

#[test]
fn buffer_usage_transfer_src() {
    assert_eq!(
        resource_state_to_buffer_usage(ResourceState::TransferSrc),
        "CopySrc",
    );
}

#[test]
fn buffer_usage_transfer_dst() {
    assert_eq!(
        resource_state_to_buffer_usage(ResourceState::TransferDst),
        "CopyDst",
    );
}

#[test]
#[should_panic(expected = "has no buffer counterpart")]
fn buffer_usage_panics_on_color_attachment() {
    resource_state_to_buffer_usage(ResourceState::ColorAttachment);
}

#[test]
#[should_panic(expected = "has no buffer counterpart")]
fn buffer_usage_panics_on_depth_stencil_attachment() {
    resource_state_to_buffer_usage(ResourceState::DepthStencilAttachment);
}

#[test]
#[should_panic(expected = "has no buffer counterpart")]
fn buffer_usage_panics_on_depth_stencil_read_only() {
    resource_state_to_buffer_usage(ResourceState::DepthStencilReadOnly);
}

#[test]
#[should_panic(expected = "has no buffer counterpart")]
fn buffer_usage_panics_on_present() {
    resource_state_to_buffer_usage(ResourceState::Present);
}

#[test]
#[should_panic(expected = "has no buffer counterpart")]
fn buffer_usage_panics_on_acceleration_structure() {
    resource_state_to_buffer_usage(ResourceState::AccelerationStructure);
}

// ===== SECTION 8 -- generate_barriers =====

#[test]
fn generate_barriers_empty_input_returns_empty() {
    let result = generate_barriers(&[], &[], &[], &[]);
    assert!(result.is_empty());
}

#[test]
fn generate_barriers_single_texture_barrier() {
    // One texture resource transitioning between two passes.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "color", 1920, 1080)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), r_tex, EdgeType::RAW)];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "write", &[r_tex]),
        // Second pass reads r_tex as ShaderRead; a compute pass suffices.
        {
            let _p = mock_pass_compute(PassIndex(1), "read", &[r_tex], &[]);
            _p
        },
    ];
    let barrier_tuples = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceState::ColorAttachment,
        ResourceState::ShaderRead,
        r_tex,
    )];

    let result = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(result.len(), 1);

    let cmd = &result[0];
    assert_eq!(cmd.texture_barriers.len(), 1);
    assert!(cmd.buffer_barriers.is_empty());

    let tb = &cmd.texture_barriers[0];
    assert_eq!(tb.resource, r_tex);
    assert_eq!(tb.before, ResourceState::ColorAttachment);
    assert_eq!(tb.after, ResourceState::ShaderRead);
}

#[test]
fn generate_barriers_single_buffer_barrier() {
    let r_buf = ResourceHandle(10);
    let resources = vec![mock_resource_buffer(r_buf, "storage", 4096)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), r_buf, EdgeType::RAW)];
    let passes = vec![
        mock_pass_compute(PassIndex(0), "write", &[], &[r_buf]),
        mock_pass_compute(PassIndex(1), "read", &[r_buf], &[]),
    ];
    let barrier_tuples = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceState::ShaderReadWrite,
        ResourceState::ShaderRead,
        r_buf,
    )];

    let result = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(result.len(), 1);

    let cmd = &result[0];
    assert!(cmd.texture_barriers.is_empty());
    assert_eq!(cmd.buffer_barriers.len(), 1);

    let bb = &cmd.buffer_barriers[0];
    assert_eq!(bb.resource, r_buf);
    assert_eq!(bb.before, ResourceState::ShaderReadWrite);
    assert_eq!(bb.after, ResourceState::ShaderRead);
}

#[test]
fn generate_barriers_mixed_texture_and_buffer() {
    // Two resources (one texture, one buffer) at two different boundaries.
    // P0→P1 transitions a texture; P1→P2 transitions a buffer.
    // Each boundary produces its own BarrierCommand with the right barrier type.
    let r_tex = ResourceHandle(1);
    let r_buf = ResourceHandle(10);
    let resources = vec![
        mock_resource_texture(r_tex, "color", 1920, 1080),
        mock_resource_buffer(r_buf, "storage", 4096),
    ];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex, EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), r_buf, EdgeType::RAW),
    ];
    let passes = vec![
        mock_pass_compute(PassIndex(0), "write_tex", &[], &[r_tex]),
        mock_pass_compute(PassIndex(1), "write_buf", &[r_tex], &[r_buf]),
        mock_pass_compute(PassIndex(2), "read_both", &[r_buf], &[]),
    ];
    let barrier_tuples = vec![
        (
            PassIndex(0),
            PassIndex(1),
            ResourceState::ShaderReadWrite,
            ResourceState::ShaderRead,
            r_tex,
        ),
        (
            PassIndex(1),
            PassIndex(2),
            ResourceState::ShaderReadWrite,
            ResourceState::ShaderRead,
            r_buf,
        ),
    ];

    let result = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(result.len(), 2, "two boundaries = two BarrierCommands");

    // First boundary (0→1): texture barrier.
    assert_eq!(result[0].texture_barriers.len(), 1);
    assert!(result[0].buffer_barriers.is_empty());

    // Second boundary (1→2): buffer barrier.
    assert!(result[1].texture_barriers.is_empty());
    assert_eq!(result[1].buffer_barriers.len(), 1);
}

#[test]
fn generate_barriers_multiple_boundaries_produce_multiple_commands() {
    // Three passes: P0 → P1 (one barrier), P1 → P2 (another barrier).
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "color", 800, 600)];
    let edges = vec![
        IrEdge::new(PassIndex(0), PassIndex(1), r_tex, EdgeType::RAW),
        IrEdge::new(PassIndex(1), PassIndex(2), r_tex, EdgeType::RAW),
    ];
    // Need realistic passes with correct access patterns for compute_barriers.
    let p0 = mock_pass_compute(PassIndex(0), "producer", &[], &[r_tex]);
    let p1 = mock_pass_compute(PassIndex(1), "middle", &[r_tex], &[r_tex]);
    let p2 = mock_pass_compute(PassIndex(2), "consumer", &[r_tex], &[]);
    let passes = vec![p0, p1, p2];
    let barrier_tuples = vec![
        (
            PassIndex(0),
            PassIndex(1),
            ResourceState::ShaderReadWrite,
            ResourceState::ShaderRead,
            r_tex,
        ),
        (
            PassIndex(1),
            PassIndex(2),
            ResourceState::ShaderReadWrite,
            ResourceState::ShaderRead,
            r_tex,
        ),
    ];

    let result = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(result.len(), 2, "two pass boundaries = two BarrierCommands");
    // Results are sorted by source pass index: (0→1) before (1→2).
    let first_textures = &result[0].texture_barriers;
    let second_textures = &result[1].texture_barriers;
    assert_eq!(first_textures.len(), 1);
    assert_eq!(second_textures.len(), 1);
}

#[test]
fn generate_barriers_unknown_resource_skips_barrier() {
    // Barrier tuple references a resource not in the resources array.
    let r_tex = ResourceHandle(1);
    let r_unknown = ResourceHandle(99);
    let resources = vec![mock_resource_texture(r_tex, "color", 800, 600)];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), r_unknown, EdgeType::RAW)];
    let passes = vec![
        mock_pass_compute(PassIndex(0), "p0", &[], &[r_unknown]),
        mock_pass_compute(PassIndex(1), "p1", &[r_unknown], &[]),
    ];
    let barrier_tuples = vec![(
        PassIndex(0),
        PassIndex(1),
        ResourceState::Uninitialized,
        ResourceState::ShaderRead,
        r_unknown,
    )];

    // The barrier tuple is skipped because r_99 has no descriptor.
    let result = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert!(result.is_empty(), "unknown resource descriptor should be skipped");
}

// ===== SECTION 9 -- Full API chain: compute_barriers -> generate_barriers =====

#[test]
fn full_chain_texture_writer_to_reader_produces_barrier_command() {
    // P0 writes R1 (color attachment), P1 reads R1 (shader read).
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "color", 800, 600)];
    let write_pass = mock_pass_graphics(PassIndex(0), "gbuffer", &[r_tex]);
    let read_pass = mock_pass_compute(PassIndex(1), "lighting", &[r_tex], &[]);
    let passes = vec![write_pass, read_pass];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), r_tex, EdgeType::RAW)];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert_eq!(barrier_tuples.len(), 1);

    let result = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(result.len(), 1);

    let cmd = &result[0];
    assert_eq!(cmd.texture_barriers.len(), 1);
    assert!(cmd.buffer_barriers.is_empty());
    assert_eq!(cmd.texture_barriers[0].before, ResourceState::ColorAttachment);
    assert_eq!(cmd.texture_barriers[0].after, ResourceState::ShaderRead);
}

#[test]
fn full_chain_no_transition_skips_barrier() {
    // Both passes read the same resource → no barrier needed.
    let r_tex = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r_tex, "shadow", 1024, 1024)];
    let p0 = mock_pass_compute(PassIndex(0), "read_a", &[r_tex], &[]);
    let p1 = mock_pass_compute(PassIndex(1), "read_b", &[r_tex], &[]);
    let passes = vec![p0, p1];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), r_tex, EdgeType::RAW)];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    // Both passes read R1 as ShaderRead; states match → no barrier.
    assert!(
        barrier_tuples.is_empty(),
        "no transition needed when both passes read with same state",
    );

    let result = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert!(result.is_empty());
}

#[test]
fn full_chain_buffer_writer_to_reader() {
    // P0 writes R1 (storage write), P1 reads R1 (shader read).
    let r_buf = ResourceHandle(10);
    let resources = vec![mock_resource_buffer(r_buf, "ssbo", 8192)];
    let write_pass = mock_pass_compute(PassIndex(0), "compute_write", &[], &[r_buf]);
    let read_pass = mock_pass_compute(PassIndex(1), "compute_read", &[r_buf], &[]);
    let passes = vec![write_pass, read_pass];
    let edges = vec![IrEdge::new(PassIndex(0), PassIndex(1), r_buf, EdgeType::RAW)];
    let order = vec![PassIndex(0), PassIndex(1)];

    let barrier_tuples = compute_barriers(&order, &passes, &edges);
    assert_eq!(barrier_tuples.len(), 1);

    let result = generate_barriers(&barrier_tuples, &passes, &edges, &resources);
    assert_eq!(result.len(), 1);

    let cmd = &result[0];
    assert!(cmd.texture_barriers.is_empty());
    assert_eq!(cmd.buffer_barriers.len(), 1);
    assert_eq!(cmd.buffer_barriers[0].before, ResourceState::ShaderReadWrite);
    assert_eq!(cmd.buffer_barriers[0].after, ResourceState::ShaderRead);
}

// ===== SECTION 10 -- ResourceState Display =====

#[test]
fn resource_state_display_all_variants() {
    let cases = vec![
        (ResourceState::Uninitialized, "Uninitialized"),
        (ResourceState::VertexBuffer, "VertexBuffer"),
        (ResourceState::IndexBuffer, "IndexBuffer"),
        (ResourceState::IndirectArgument, "IndirectArgument"),
        (ResourceState::ColorAttachment, "ColorAttachment"),
        (ResourceState::DepthStencilAttachment, "DepthStencilAttachment"),
        (ResourceState::DepthStencilReadOnly, "DepthStencilReadOnly"),
        (ResourceState::ShaderRead, "ShaderRead"),
        (ResourceState::ShaderReadWrite, "ShaderReadWrite"),
        (ResourceState::TransferSrc, "TransferSrc"),
        (ResourceState::TransferDst, "TransferDst"),
        (ResourceState::AccelerationStructure, "AccelerationStructure"),
        (ResourceState::Present, "Present"),
    ];
    for (state, expected) in &cases {
        assert_eq!(&format!("{}", state), expected, "Display for {:?}", state);
        assert_eq!(&format!("{:?}", state), expected, "Debug for {:?}", state);
    }
}
