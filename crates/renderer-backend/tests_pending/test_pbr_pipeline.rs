//! Tests for T-MAT-3.4: PBR Pipeline Integration
//!
//! This test module verifies the PBR (Physically Based Rendering) pipeline
//! integration, including:
//! - PbrVertex and PbrVertexTangent struct layouts
//! - Vertex buffer layout correctness
//! - Pipeline creation with various configurations
//! - Bind group layout compatibility
//! - Default PBR shader compilation

use renderer_backend::pipeline::{
    pbr_vertex_buffer_layout, pbr_vertex_tangent_buffer_layout,
    PbrVertex, PbrVertexTangent, PipelineTable, PBR_SHADER_SRC,
};

// ---------------------------------------------------------------------------
// Vertex struct tests
// ---------------------------------------------------------------------------

#[test]
fn test_pbr_vertex_size() {
    // PbrVertex should be exactly 32 bytes:
    // - position: 3 * f32 = 12 bytes
    // - normal:   3 * f32 = 12 bytes
    // - uv:       2 * f32 =  8 bytes
    // Total: 32 bytes
    assert_eq!(std::mem::size_of::<PbrVertex>(), 32);
}

#[test]
fn test_pbr_vertex_tangent_size() {
    // PbrVertexTangent should be exactly 48 bytes:
    // - position: 3 * f32 = 12 bytes
    // - normal:   3 * f32 = 12 bytes
    // - uv:       2 * f32 =  8 bytes
    // - tangent:  4 * f32 = 16 bytes
    // Total: 48 bytes
    assert_eq!(std::mem::size_of::<PbrVertexTangent>(), 48);
}

#[test]
fn test_pbr_vertex_alignment() {
    // Vertices should be aligned to 4 bytes (f32 alignment)
    assert_eq!(std::mem::align_of::<PbrVertex>(), 4);
    assert_eq!(std::mem::align_of::<PbrVertexTangent>(), 4);
}

#[test]
fn test_pbr_vertex_bytemuck() {
    // Verify bytemuck Pod/Zeroable traits work
    let vertex = PbrVertex::new([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.5, 0.5]);
    let bytes: &[u8] = bytemuck::bytes_of(&vertex);
    assert_eq!(bytes.len(), 32);

    // Verify round-trip
    let vertex_back: PbrVertex = *bytemuck::from_bytes(bytes);
    assert_eq!(vertex_back.position, vertex.position);
    assert_eq!(vertex_back.normal, vertex.normal);
    assert_eq!(vertex_back.uv, vertex.uv);
}

#[test]
fn test_pbr_vertex_tangent_bytemuck() {
    let vertex = PbrVertexTangent::new(
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5],
        [1.0, 0.0, 0.0, 1.0],
    );
    let bytes: &[u8] = bytemuck::bytes_of(&vertex);
    assert_eq!(bytes.len(), 48);

    let vertex_back: PbrVertexTangent = *bytemuck::from_bytes(bytes);
    assert_eq!(vertex_back.position, vertex.position);
    assert_eq!(vertex_back.normal, vertex.normal);
    assert_eq!(vertex_back.uv, vertex.uv);
    assert_eq!(vertex_back.tangent, vertex.tangent);
}

#[test]
fn test_pbr_vertex_slice_cast() {
    // Test casting a slice of vertices to bytes
    let vertices = [
        PbrVertex::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
        PbrVertex::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0]),
        PbrVertex::new([0.5, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0]),
    ];
    let bytes: &[u8] = bytemuck::cast_slice(&vertices);
    assert_eq!(bytes.len(), 3 * 32);
}

// ---------------------------------------------------------------------------
// Vertex buffer layout tests
// ---------------------------------------------------------------------------

#[test]
fn test_pbr_vertex_buffer_layout_stride() {
    let layout = pbr_vertex_buffer_layout();
    assert_eq!(layout.array_stride, 32);
    assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
}

#[test]
fn test_pbr_vertex_buffer_layout_attributes() {
    let layout = pbr_vertex_buffer_layout();
    assert_eq!(layout.attributes.len(), 3);

    // Position: vec3<f32> at location 0, offset 0
    assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Float32x3);
    assert_eq!(layout.attributes[0].offset, 0);
    assert_eq!(layout.attributes[0].shader_location, 0);

    // Normal: vec3<f32> at location 1, offset 12
    assert_eq!(layout.attributes[1].format, wgpu::VertexFormat::Float32x3);
    assert_eq!(layout.attributes[1].offset, 12);
    assert_eq!(layout.attributes[1].shader_location, 1);

    // UV: vec2<f32> at location 2, offset 24
    assert_eq!(layout.attributes[2].format, wgpu::VertexFormat::Float32x2);
    assert_eq!(layout.attributes[2].offset, 24);
    assert_eq!(layout.attributes[2].shader_location, 2);
}

#[test]
fn test_pbr_vertex_tangent_buffer_layout_stride() {
    let layout = pbr_vertex_tangent_buffer_layout();
    assert_eq!(layout.array_stride, 48);
    assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
}

#[test]
fn test_pbr_vertex_tangent_buffer_layout_attributes() {
    let layout = pbr_vertex_tangent_buffer_layout();
    assert_eq!(layout.attributes.len(), 4);

    // Position: vec3<f32> at location 0, offset 0
    assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Float32x3);
    assert_eq!(layout.attributes[0].offset, 0);
    assert_eq!(layout.attributes[0].shader_location, 0);

    // Normal: vec3<f32> at location 1, offset 12
    assert_eq!(layout.attributes[1].format, wgpu::VertexFormat::Float32x3);
    assert_eq!(layout.attributes[1].offset, 12);
    assert_eq!(layout.attributes[1].shader_location, 1);

    // UV: vec2<f32> at location 2, offset 24
    assert_eq!(layout.attributes[2].format, wgpu::VertexFormat::Float32x2);
    assert_eq!(layout.attributes[2].offset, 24);
    assert_eq!(layout.attributes[2].shader_location, 2);

    // Tangent: vec4<f32> at location 3, offset 32
    assert_eq!(layout.attributes[3].format, wgpu::VertexFormat::Float32x4);
    assert_eq!(layout.attributes[3].offset, 32);
    assert_eq!(layout.attributes[3].shader_location, 3);
}

// ---------------------------------------------------------------------------
// PBR shader tests
// ---------------------------------------------------------------------------

#[test]
fn test_pbr_shader_parses() {
    // Verify the default PBR shader is valid WGSL
    let module = naga::front::wgsl::parse_str(PBR_SHADER_SRC);
    assert!(
        module.is_ok(),
        "PBR shader should parse: {:?}",
        module.err()
    );
}

#[test]
fn test_pbr_shader_has_entry_points() {
    let module = naga::front::wgsl::parse_str(PBR_SHADER_SRC).expect("parse");

    // Find vertex and fragment entry points
    let has_vs_main = module
        .entry_points
        .iter()
        .any(|ep| ep.name == "vs_main" && ep.stage == naga::ShaderStage::Vertex);
    let has_fs_main = module
        .entry_points
        .iter()
        .any(|ep| ep.name == "fs_main" && ep.stage == naga::ShaderStage::Fragment);

    assert!(has_vs_main, "PBR shader must have vs_main vertex entry point");
    assert!(has_fs_main, "PBR shader must have fs_main fragment entry point");
}

#[test]
fn test_pbr_shader_validates() {
    let module = naga::front::wgsl::parse_str(PBR_SHADER_SRC).expect("parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&module);
    assert!(
        info.is_ok(),
        "PBR shader should validate: {:?}",
        info.err()
    );
}

// ---------------------------------------------------------------------------
// GPU pipeline tests (require wgpu device)
// ---------------------------------------------------------------------------

fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: true,
    }))?;
    Some(
        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("device creation"),
    )
}

#[test]
fn test_pbr_pipeline_creates() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("PBR Test Shader"),
        source: wgpu::ShaderSource::Wgsl(PBR_SHADER_SRC.into()),
    });

    let mut table = PipelineTable::new();
    let result = table.create_pbr_pipeline(
        &device,
        1,                               // id
        &shader,                         // vertex shader
        &shader,                         // fragment shader (same module)
        wgpu::TextureFormat::Rgba8Unorm, // surface format
        None,                            // no depth
        1,                               // no MSAA
    );

    assert!(result.is_ok(), "PBR pipeline should create successfully");
    assert_eq!(result.unwrap(), 1);
    assert!(table.get(1).is_some());
}

#[test]
fn test_pbr_pipeline_with_depth() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("PBR Test Shader"),
        source: wgpu::ShaderSource::Wgsl(PBR_SHADER_SRC.into()),
    });

    let mut table = PipelineTable::new();
    let result = table.create_pbr_pipeline(
        &device,
        2,
        &shader,
        &shader,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::TextureFormat::Depth32Float), // with depth
        1,
    );

    assert!(result.is_ok(), "PBR pipeline with depth should create");
    let cached = table.get(2).expect("pipeline should be cached");
    assert_eq!(cached.id, 2);
}

#[test]
fn test_pbr_pipeline_with_msaa() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("PBR Test Shader"),
        source: wgpu::ShaderSource::Wgsl(PBR_SHADER_SRC.into()),
    });

    let mut table = PipelineTable::new();
    let result = table.create_pbr_pipeline(
        &device,
        3,
        &shader,
        &shader,
        wgpu::TextureFormat::Bgra8Unorm, // different format
        Some(wgpu::TextureFormat::Depth24Plus),
        4, // 4x MSAA
    );

    assert!(result.is_ok(), "PBR pipeline with 4x MSAA should create");
}

#[test]
fn test_pbr_pipeline_bind_group_layout() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("PBR Test Shader"),
        source: wgpu::ShaderSource::Wgsl(PBR_SHADER_SRC.into()),
    });

    let mut table = PipelineTable::new();
    table
        .create_pbr_pipeline(
            &device,
            4,
            &shader,
            &shader,
            wgpu::TextureFormat::Rgba8Unorm,
            None,
            1,
        )
        .expect("create pipeline");

    let cached = table.get(4).expect("pipeline should exist");

    // Verify we can create a bind group with the layout
    let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Camera Buffer"),
        size: 80, // mat4x4 + vec4
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let material_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Material Buffer"),
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Light Buffer"),
        size: 48,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    // This should not panic if bind group layout is correct
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Test Bind Group"),
        layout: &cached.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: material_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: light_buffer.as_entire_binding(),
            },
        ],
    });

    // If we get here without panicking, the bind group layout is valid
    let _ = bind_group;
}

#[test]
fn test_pbr_pipeline_multiple_ids() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("PBR Test Shader"),
        source: wgpu::ShaderSource::Wgsl(PBR_SHADER_SRC.into()),
    });

    let mut table = PipelineTable::new();

    // Create multiple pipelines with different IDs
    for i in 10..15 {
        let result = table.create_pbr_pipeline(
            &device,
            i,
            &shader,
            &shader,
            wgpu::TextureFormat::Rgba8Unorm,
            None,
            1,
        );
        assert!(result.is_ok(), "Pipeline {} should create", i);
    }

    assert_eq!(table.len(), 5);

    for i in 10..15 {
        assert!(table.get(i).is_some(), "Pipeline {} should exist", i);
    }
}

#[test]
fn test_pbr_pipeline_replaces_existing() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("PBR Test Shader"),
        source: wgpu::ShaderSource::Wgsl(PBR_SHADER_SRC.into()),
    });

    let mut table = PipelineTable::new();

    // Create pipeline with ID 100
    table
        .create_pbr_pipeline(
            &device,
            100,
            &shader,
            &shader,
            wgpu::TextureFormat::Rgba8Unorm,
            None,
            1,
        )
        .expect("first create");

    // Create another pipeline with same ID (should replace)
    table
        .create_pbr_pipeline(
            &device,
            100,
            &shader,
            &shader,
            wgpu::TextureFormat::Bgra8Unorm, // different format
            Some(wgpu::TextureFormat::Depth32Float),
            1,
        )
        .expect("second create");

    // Table should still have only 1 pipeline
    assert_eq!(table.len(), 1);
    assert!(table.get(100).is_some());
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_pbr_vertex_data_triangle() {
    // Create a simple triangle with PBR vertices
    let vertices = [
        PbrVertex::new([0.0, 0.5, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0]),
        PbrVertex::new([-0.5, -0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]),
        PbrVertex::new([0.5, -0.5, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0]),
    ];

    // Verify data layout
    let bytes: &[u8] = bytemuck::cast_slice(&vertices);
    assert_eq!(bytes.len(), 96); // 3 vertices * 32 bytes

    // Verify first vertex position
    let first_pos: [f32; 3] = *bytemuck::from_bytes(&bytes[0..12]);
    assert_eq!(first_pos, [0.0, 0.5, 0.0]);

    // Verify first vertex normal
    let first_normal: [f32; 3] = *bytemuck::from_bytes(&bytes[12..24]);
    assert_eq!(first_normal, [0.0, 0.0, 1.0]);

    // Verify first vertex UV
    let first_uv: [f32; 2] = *bytemuck::from_bytes(&bytes[24..32]);
    assert_eq!(first_uv, [0.5, 0.0]);
}

#[test]
fn test_pbr_pipeline_table_new_is_empty() {
    let table = PipelineTable::new();
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
}
