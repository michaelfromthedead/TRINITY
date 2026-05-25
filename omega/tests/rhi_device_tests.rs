//! Device creation, adapter enumeration, and feature query tests.
//!
//! Mirrors tests/platform/rhi/test_device.py.

mod common;

use common::*;

#[test]
fn test_adapter_enumerate_returns_different_types() {
    let adapters = MockAdapter::enumerate();
    assert_eq!(adapters.len(), 3);

    // Verify adapter types differ
    let types: std::collections::HashSet<AdapterType> =
        adapters.iter().map(|a| a.info().adapter_type).collect();
    assert!(types.len() > 1);
}

#[test]
fn test_adapter_info_discrete() {
    let adapter = MockAdapter::new(AdapterType::Discrete);
    let info = adapter.info();

    assert!(info.name.starts_with("Mock Adapter"));
    assert_eq!(info.adapter_type, AdapterType::Discrete);
    assert!(info.dedicated_video_memory > 0);
    assert_eq!(info.vendor_id, NULL_VENDOR_ID);
    assert_eq!(info.device_id, NULL_DEVICE_ID);
}

#[test]
fn test_adapter_query_features_discrete() {
    let adapter = MockAdapter::new(AdapterType::Discrete);
    let features = adapter.query_features();

    assert!(features.ray_tracing);
    assert!(features.mesh_shaders);
    assert!(features.bindless);
    assert!(features.compute);
    assert!(features.max_texture_size > 0);
}

#[test]
fn test_adapter_query_features_integrated() {
    let adapter = MockAdapter::new(AdapterType::Integrated);
    let features = adapter.query_features();

    assert!(!features.ray_tracing);
    assert!(!features.mesh_shaders);
    assert!(features.bindless);
    assert!(features.compute);
}

#[test]
fn test_adapter_query_format_support() {
    let adapter = MockAdapter::new(AdapterType::Software);
    let support = adapter.query_format_support(Format::RGBA8Unorm);

    assert!(support.renderable);
    assert!(support.filterable);
    assert!(support.blendable);
    assert!(support.storage);
    assert!(support.multisample);
}

#[test]
fn test_device_create_with_defaults() {
    let adapter = MockAdapter::new(AdapterType::Software);
    let device = MockDevice::create(adapter);

    // Verify queues are available
    let gfx = device.get_queue(QueueType::Graphics);
    assert_eq!(gfx.queue_type(), QueueType::Graphics);
}

#[test]
fn test_device_create_with_config() {
    let adapter = MockAdapter::new(AdapterType::Discrete);
    let config = DeviceConfig {
        enable_debug: true,
        enable_validation: true,
    };
    let device = MockDevice::create_with_config(adapter, config);

    assert!(device.debug_enabled());
    assert!(device.validation_enabled());
}

#[test]
fn test_device_get_all_queue_types() {
    let device = create_test_device();

    let gfx = device.get_queue(QueueType::Graphics);
    let compute = device.get_queue(QueueType::Compute);
    let transfer = device.get_queue(QueueType::Transfer);

    assert_eq!(gfx.queue_type(), QueueType::Graphics);
    assert_eq!(compute.queue_type(), QueueType::Compute);
    assert_eq!(transfer.queue_type(), QueueType::Transfer);
}

#[test]
fn test_device_queue_caching_returns_same_instance() {
    let device = create_test_device();
    let q1 = device.get_queue(QueueType::Graphics);
    let q2 = device.get_queue(QueueType::Graphics);

    // Same underlying queue is returned (same Arc)
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.draw(3, 1, 0, 0);
    cmd.end();
    q1.submit(&[cmd]);
    assert_eq!(q2.submitted_count(), 1);
}

#[test]
fn test_device_create_buffer() {
    let device = create_test_device();
    let desc = BufferDesc {
        size: 1024,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        memory_type: MemoryType::Default,
        stride: 0,
    };
    let buffer = device.create_buffer(desc);

    assert!(buffer.is_valid());
    assert_eq!(buffer.desc().size, 1024);
    assert!(buffer.handle() > 0);
}

#[test]
fn test_device_create_texture() {
    let device = create_test_device();
    let desc = TextureDesc {
        ty: TextureType::Texture2D,
        format: Format::RGBA8Unorm,
        width: 1024,
        height: 768,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_RESOURCE,
    };
    let texture = device.create_texture(desc);

    assert!(texture.is_valid());
    assert_eq!(texture.desc().width, 1024);
    assert_eq!(texture.desc().height, 768);
}

#[test]
fn test_device_create_sampler() {
    let device = create_test_device();
    let desc = SamplerDesc {
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        ..Default::default()
    };
    let sampler = device.create_sampler(desc);

    assert!(sampler.is_valid());
    assert!(sampler.handle() > 0);
}

#[test]
fn test_device_create_graphics_pipeline() {
    let device = create_test_device();
    let vs = ShaderDesc {
        stage: ShaderStage::Vertex,
        source: b"vs_code".to_vec(),
        entry_point: "vs_main".into(),
    };
    let ps = ShaderDesc {
        stage: ShaderStage::Pixel,
        source: b"ps_code".to_vec(),
        entry_point: "ps_main".into(),
    };
    let pipeline_desc = GraphicsPipelineDesc {
        vertex_shader: Some(vs),
        pixel_shader: Some(ps),
        ..Default::default()
    };
    let pipeline = device.create_graphics_pipeline(&pipeline_desc);

    assert!(pipeline.is_valid());
    assert_eq!(pipeline.pipeline_type(), PipelineType::Graphics);
}

#[test]
fn test_device_create_compute_pipeline() {
    let device = create_test_device();
    let cs = ShaderDesc {
        stage: ShaderStage::Compute,
        source: b"cs_code".to_vec(),
        entry_point: "cs_main".into(),
    };
    let pipeline_desc = ComputePipelineDesc {
        compute_shader: Some(cs),
    };
    let pipeline = device.create_compute_pipeline(&pipeline_desc);

    assert!(pipeline.is_valid());
    assert_eq!(pipeline.pipeline_type(), PipelineType::Compute);
}

#[test]
fn test_device_wait_idle_does_not_hang() {
    let device = create_test_device();
    device.wait_idle();
}

#[test]
fn test_device_shutdown() {
    let mut device = create_test_device();
    assert!(!device.is_shutdown());
    device.shutdown();
    assert!(device.is_shutdown());
}

#[test]
fn test_discrete_adapter_has_ray_tracing() {
    let adapter = MockAdapter::new(AdapterType::Discrete);
    assert!(adapter.query_features().ray_tracing);
}

#[test]
fn test_integrated_adapter_lacks_ray_tracing() {
    let adapter = MockAdapter::new(AdapterType::Integrated);
    assert!(!adapter.query_features().ray_tracing);
}

#[test]
fn test_software_adapter_lacks_ray_tracing() {
    let adapter = MockAdapter::new(AdapterType::Software);
    assert!(!adapter.query_features().ray_tracing);
}

#[test]
fn test_adapter_info_integrated() {
    let adapter = MockAdapter::new(AdapterType::Integrated);
    let info = adapter.info();
    assert_eq!(info.adapter_type, AdapterType::Integrated);
    assert!(info.shared_system_memory > 0);
}

#[test]
fn test_adapter_type_unique_names() {
    let discrete = MockAdapter::new(AdapterType::Discrete);
    let integrated = MockAdapter::new(AdapterType::Integrated);
    let software = MockAdapter::new(AdapterType::Software);

    assert_ne!(discrete.info().name, integrated.info().name);
    assert_ne!(integrated.info().name, software.info().name);
}
