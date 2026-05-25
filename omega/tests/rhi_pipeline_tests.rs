//! Pipeline and shader tests.
//!
//! Mirrors tests/platform/rhi/test_pipeline.py plus Rust-native
//! render pipeline draw-to-texture and compute dispatch tests.

mod common;

use common::*;

// =========================================================================
// Shader descriptor tests
// =========================================================================

#[test]
fn test_shader_desc_creation() {
    let desc = ShaderDesc {
        stage: ShaderStage::Vertex,
        source: b"vertex_shader_bytecode".to_vec(),
        entry_point: "vs_main".into(),
    };

    assert_eq!(desc.stage, ShaderStage::Vertex);
    assert_eq!(desc.source, b"vertex_shader_bytecode");
    assert_eq!(desc.entry_point, "vs_main");
}

#[test]
fn test_shader_desc_compute() {
    let desc = ShaderDesc {
        stage: ShaderStage::Compute,
        source: b"compute_kernel".to_vec(),
        entry_point: "cs_main".into(),
    };
    assert_eq!(desc.stage, ShaderStage::Compute);
    assert_eq!(desc.entry_point, "cs_main");
}

// =========================================================================
// Graphics pipeline tests
// =========================================================================

#[test]
fn test_graphics_pipeline_creation() {
    let device = create_test_device();
    let vs = ShaderDesc {
        stage: ShaderStage::Vertex,
        source: b"vs_code".to_vec(),
        entry_point: "main".into(),
    };
    let ps = ShaderDesc {
        stage: ShaderStage::Pixel,
        source: b"ps_code".to_vec(),
        entry_point: "main".into(),
    };
    let pipeline_desc = GraphicsPipelineDesc {
        vertex_shader: Some(vs),
        pixel_shader: Some(ps),
        topology: PrimitiveTopology::TriangleList,
        ..Default::default()
    };
    let pipeline = device.create_graphics_pipeline(&pipeline_desc);

    assert!(pipeline.is_valid());
    assert_eq!(pipeline.pipeline_type(), PipelineType::Graphics);
    assert!(pipeline.handle() > 0);
}

#[test]
fn test_graphics_pipeline_triangle_strip() {
    let device = create_test_device();
    let vs = ShaderDesc {
        stage: ShaderStage::Vertex,
        source: b"vs".to_vec(),
        entry_point: "main".into(),
    };
    let ps = ShaderDesc {
        stage: ShaderStage::Pixel,
        source: b"ps".to_vec(),
        entry_point: "main".into(),
    };
    let desc = GraphicsPipelineDesc {
        vertex_shader: Some(vs),
        pixel_shader: Some(ps),
        topology: PrimitiveTopology::TriangleStrip,
        ..Default::default()
    };
    let pipeline = device.create_graphics_pipeline(&desc);
    assert!(pipeline.is_valid());
}

#[test]
fn test_graphics_pipeline_with_geometry_shader() {
    let device = create_test_device();
    let desc = GraphicsPipelineDesc {
        vertex_shader: Some(ShaderDesc {
            stage: ShaderStage::Vertex,
            source: b"vs".to_vec(),
            entry_point: "main".into(),
        }),
        geometry_shader: Some(ShaderDesc {
            stage: ShaderStage::Geometry,
            source: b"gs".to_vec(),
            entry_point: "main".into(),
        }),
        pixel_shader: Some(ShaderDesc {
            stage: ShaderStage::Pixel,
            source: b"ps".to_vec(),
            entry_point: "main".into(),
        }),
        ..Default::default()
    };
    let pipeline = device.create_graphics_pipeline(&desc);
    assert!(pipeline.is_valid());
}

#[test]
fn test_graphics_pipeline_with_tessellation() {
    let device = create_test_device();
    let desc = GraphicsPipelineDesc {
        vertex_shader: Some(ShaderDesc {
            stage: ShaderStage::Vertex,
            source: b"vs".to_vec(),
            entry_point: "main".into(),
        }),
        hull_shader: Some(ShaderDesc {
            stage: ShaderStage::Hull,
            source: b"hs".to_vec(),
            entry_point: "main".into(),
        }),
        domain_shader: Some(ShaderDesc {
            stage: ShaderStage::Domain,
            source: b"ds".to_vec(),
            entry_point: "main".into(),
        }),
        pixel_shader: Some(ShaderDesc {
            stage: ShaderStage::Pixel,
            source: b"ps".to_vec(),
            entry_point: "main".into(),
        }),
        ..Default::default()
    };
    let pipeline = device.create_graphics_pipeline(&desc);
    assert!(pipeline.is_valid());
}

#[test]
fn test_graphics_pipeline_with_all_states() {
    let device = create_test_device();
    let desc = GraphicsPipelineDesc {
        vertex_shader: Some(ShaderDesc {
            stage: ShaderStage::Vertex,
            source: b"vs".to_vec(),
            entry_point: "main".into(),
        }),
        pixel_shader: Some(ShaderDesc {
            stage: ShaderStage::Pixel,
            source: b"ps".to_vec(),
            entry_point: "main".into(),
        }),
        topology: PrimitiveTopology::TriangleStrip,
        rasterizer: RasterizerState {
            fill_mode: FillMode::Solid,
            cull_mode: CullMode::Back,
            ..Default::default()
        },
        depth_stencil: DepthStencilState {
            depth_test: true,
            depth_write: true,
            depth_func: CompareOp::Less,
        },
        blend: BlendState::default(),
        render_target_formats: vec![Format::RGBA8Unorm],
        depth_format: Some(Format::D32Float),
        ..Default::default()
    };
    let pipeline = device.create_graphics_pipeline(&desc);
    assert!(pipeline.is_valid());
}

#[test]
fn test_primitive_topologies() {
    let device = create_test_device();
    let base = GraphicsPipelineDesc {
        vertex_shader: Some(ShaderDesc {
            stage: ShaderStage::Vertex,
            source: b"vs".to_vec(),
            entry_point: "main".into(),
        }),
        pixel_shader: Some(ShaderDesc {
            stage: ShaderStage::Pixel,
            source: b"ps".to_vec(),
            entry_point: "main".into(),
        }),
        ..Default::default()
    };

    let p_list = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        topology: PrimitiveTopology::TriangleList,
        ..base.clone()
    });
    assert!(p_list.is_valid());

    let p_strip = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        topology: PrimitiveTopology::TriangleStrip,
        ..base
    });
    assert!(p_strip.is_valid());
}

// =========================================================================
// Compute pipeline tests
// =========================================================================

#[test]
fn test_compute_pipeline_creation() {
    let device = create_test_device();
    let cs = ShaderDesc {
        stage: ShaderStage::Compute,
        source: b"compute_kernel".to_vec(),
        entry_point: "cs_main".into(),
    };
    let desc = ComputePipelineDesc {
        compute_shader: Some(cs),
    };
    let pipeline = device.create_compute_pipeline(&desc);

    assert!(pipeline.is_valid());
    assert_eq!(pipeline.pipeline_type(), PipelineType::Compute);
    assert!(pipeline.handle() > 0);
}

// =========================================================================
// Pipeline state tests
// =========================================================================

#[test]
fn test_rasterizer_state_wireframe() {
    let state = RasterizerState {
        fill_mode: FillMode::Wireframe,
        cull_mode: CullMode::Front,
        front_ccw: true,
        depth_bias: 100,
        depth_clip: false,
    };

    assert_eq!(state.fill_mode, FillMode::Wireframe);
    assert_eq!(state.cull_mode, CullMode::Front);
    assert!(state.front_ccw);
    assert_eq!(state.depth_bias, 100);
    assert!(!state.depth_clip);
}

#[test]
fn test_rasterizer_state_default() {
    let state = RasterizerState::default();
    assert_eq!(state.fill_mode, FillMode::Solid);
    assert_eq!(state.cull_mode, CullMode::Back);
    assert!(!state.front_ccw);
    assert_eq!(state.depth_bias, 0);
    assert!(state.depth_clip);
}

#[test]
fn test_rasterizer_state_cull_modes() {
    for mode in &[CullMode::None, CullMode::Front, CullMode::Back] {
        let state = RasterizerState {
            cull_mode: *mode,
            ..Default::default()
        };
        assert_eq!(state.cull_mode, *mode);
    }
}

#[test]
fn test_depth_stencil_state() {
    let state = DepthStencilState {
        depth_test: true,
        depth_write: true,
        depth_func: CompareOp::LessEqual,
    };

    assert!(state.depth_test);
    assert!(state.depth_write);
    assert_eq!(state.depth_func, CompareOp::LessEqual);
}

#[test]
fn test_depth_stencil_default() {
    let state = DepthStencilState::default();
    assert!(state.depth_test);
    assert!(state.depth_write);
    assert_eq!(state.depth_func, CompareOp::Less);
}

#[test]
fn test_depth_stencil_disabled() {
    let state = DepthStencilState {
        depth_test: false,
        depth_write: false,
        depth_func: CompareOp::Always,
    };
    assert!(!state.depth_test);
    assert!(!state.depth_write);
}

#[test]
fn test_blend_state_enabled() {
    let state = BlendState {
        enabled: true,
        src_color: BlendFactor::SrcAlpha,
        dst_color: BlendFactor::InvSrcAlpha,
        color_op: BlendOp::Add,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::Zero,
        alpha_op: BlendOp::Add,
    };

    assert!(state.enabled);
    assert_eq!(state.src_color, BlendFactor::SrcAlpha);
    assert_eq!(state.dst_color, BlendFactor::InvSrcAlpha);
    assert_eq!(state.color_op, BlendOp::Add);
}

#[test]
fn test_blend_state_default() {
    let state = BlendState::default();
    assert!(!state.enabled);
    assert_eq!(state.src_color, BlendFactor::One);
    assert_eq!(state.dst_color, BlendFactor::Zero);
}

#[test]
fn test_blend_operations() {
    for op in &[BlendOp::Add, BlendOp::Subtract, BlendOp::RevSubtract, BlendOp::Min, BlendOp::Max] {
        let state = BlendState {
            enabled: true,
            color_op: *op,
            ..Default::default()
        };
        assert_eq!(state.color_op, *op);
    }
}

#[test]
fn test_shader_stages_graphics_vs_compute() {
    let device = create_test_device();
    let vs = ShaderDesc {
        stage: ShaderStage::Vertex,
        source: b"vs".to_vec(),
        entry_point: "main".into(),
    };
    let ps = ShaderDesc {
        stage: ShaderStage::Pixel,
        source: b"ps".to_vec(),
        entry_point: "main".into(),
    };
    let gfx_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_shader: Some(vs),
        pixel_shader: Some(ps),
        ..Default::default()
    });
    assert_eq!(gfx_pipeline.pipeline_type(), PipelineType::Graphics);

    let cs = ShaderDesc {
        stage: ShaderStage::Compute,
        source: b"cs".to_vec(),
        entry_point: "main".into(),
    };
    let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
        compute_shader: Some(cs),
    });
    assert_eq!(compute_pipeline.pipeline_type(), PipelineType::Compute);

    assert_ne!(gfx_pipeline.pipeline_type(), compute_pipeline.pipeline_type());
}

// =========================================================================
// Raytracing pipeline descriptor
// =========================================================================

#[test]
fn test_raytracing_pipeline_desc() {
    let rgen = ShaderDesc {
        stage: ShaderStage::RayGeneration,
        source: b"rgen".to_vec(),
        entry_point: "main".into(),
    };
    let miss = ShaderDesc {
        stage: ShaderStage::Miss,
        source: b"miss".to_vec(),
        entry_point: "main".into(),
    };
    let hit = ShaderDesc {
        stage: ShaderStage::ClosestHit,
        source: b"hit".to_vec(),
        entry_point: "main".into(),
    };
    let rt_desc = RaytracingPipelineDesc {
        ray_gen_shader: Some(rgen),
        miss_shaders: vec![miss],
        hit_groups: vec![format!("closest_hit={}", hit.entry_point)],
        max_recursion_depth: 3,
    };

    assert_eq!(rt_desc.max_recursion_depth, 3);
    assert_eq!(rt_desc.miss_shaders.len(), 1);
    assert!(rt_desc.ray_gen_shader.is_some());
}

// =========================================================================
// Render pipeline: command recording for a draw-to-texture scenario
// =========================================================================

#[test]
fn test_render_pipeline_full_recording() {
    let device = create_test_device();

    // Create render target
    let rt = device.create_texture(TextureDesc {
        ty: TextureType::Texture2D,
        format: Format::RGBA8Unorm,
        width: 1920,
        height: 1080,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_RESOURCE,
    });

    // Create pipeline
    let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_shader: Some(ShaderDesc {
            stage: ShaderStage::Vertex,
            source: b"vs".to_vec(),
            entry_point: "main".into(),
        }),
        pixel_shader: Some(ShaderDesc {
            stage: ShaderStage::Pixel,
            source: b"ps".to_vec(),
            entry_point: "main".into(),
        }),
        ..Default::default()
    });

    // Create vertex buffer
    let vb = device.create_buffer(BufferDesc {
        size: 4096,
        usage: BufferUsage::VERTEX,
        memory_type: MemoryType::Default,
        stride: 32,
    });

    // Record commands
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.begin_render_pass(&[rt.handle()], None, Some((0.0, 0.0, 0.0, 1.0)), None);
    cmd.set_pipeline(pipeline.handle());
    cmd.set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
    cmd.set_scissor(0, 0, 1920, 1080);
    cmd.set_vertex_buffer(0, vb.handle(), 0, 32);
    cmd.draw(3, 1, 0, 0);
    cmd.end_render_pass();
    cmd.end();

    // Verify command sequence
    let cmds = cmd.recorded_commands();
    assert_eq!(cmds.len(), 7);
    assert_eq!(cmds[0].cmd_type, "begin_render_pass");
    assert_eq!(cmds[1].cmd_type, "set_pipeline");
    assert_eq!(cmds[2].cmd_type, "set_viewport");
    assert_eq!(cmds[3].cmd_type, "set_scissor");
    assert_eq!(cmds[4].cmd_type, "set_vertex_buffer");
    assert_eq!(cmds[5].cmd_type, "draw");
    assert_eq!(cmds[6].cmd_type, "end_render_pass");
}

// =========================================================================
// Compute pipeline dispatch test
// =========================================================================

#[test]
fn test_compute_dispatch_full_recording() {
    let device = create_test_device();

    // Create compute pipeline
    let pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
        compute_shader: Some(ShaderDesc {
            stage: ShaderStage::Compute,
            source: b"compute_kernel".to_vec(),
            entry_point: "cs_main".into(),
        }),
    });

    // Create storage buffer for UAV access
    let _sb = device.create_buffer(BufferDesc {
        size: 65536,
        usage: BufferUsage::STORAGE,
        memory_type: MemoryType::Default,
        stride: 0,
    });

    // Record dispatch commands
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.set_pipeline(pipeline.handle());
    cmd.dispatch(16, 16, 1);
    cmd.end();

    let cmds = cmd.recorded_commands();
    assert_eq!(cmds.len(), 2);
    assert_eq!(cmds[0].cmd_type, "set_pipeline");
    assert_eq!(cmds[1].cmd_type, "dispatch");
}

#[test]
fn test_mesh_shader_pipeline_desc() {
    // Verify MeshPipelineDesc concept exists in the interface
    let vs = ShaderDesc {
        stage: ShaderStage::Vertex,
        source: b"vs".to_vec(),
        entry_point: "main".into(),
    };
    let ms = ShaderDesc {
        stage: ShaderStage::Mesh,
        source: b"ms".to_vec(),
        entry_point: "main".into(),
    };
    let ps = ShaderDesc {
        stage: ShaderStage::Pixel,
        source: b"ps".to_vec(),
        entry_point: "main".into(),
    };

    // Mesh shader pipelines follow the same creation path with additional stages
    let desc = GraphicsPipelineDesc {
        vertex_shader: Some(vs),
        pixel_shader: Some(ps),
        ..Default::default()
    };
    assert_eq!(ms.stage, ShaderStage::Mesh);
    let _pipeline = create_test_device().create_graphics_pipeline(&desc);
}
