"""Tests for RHI device and adapter."""
import pytest
from engine.platform.rhi import (
    Adapter, AdapterType, Device, DeviceConfig, FeatureSupport,
    NullAdapter, NullDevice, QueueType, Format
)


def test_adapter_enumerate():
    """Test adapter enumeration returns different adapter types."""
    adapters = NullAdapter.enumerate()
    assert len(adapters) == 3
    assert isinstance(adapters[0], Adapter)

    # Verify adapter types differ
    types = {a.info().adapter_type for a in adapters}
    assert len(types) > 1  # Should have discrete, integrated, software


def test_adapter_info():
    """Test adapter info retrieval."""
    adapter = NullAdapter(AdapterType.DISCRETE)
    info = adapter.info()

    assert info.name.startswith("Null Adapter")
    assert info.adapter_type == AdapterType.DISCRETE
    assert info.dedicated_video_memory > 0
    assert info.vendor_id == 0x0000
    assert info.device_id == 0x0000


def test_adapter_query_features():
    """Test feature support query."""
    discrete_adapter = NullAdapter(AdapterType.DISCRETE)
    discrete_features = discrete_adapter.query_features()

    assert isinstance(discrete_features, FeatureSupport)
    assert discrete_features.ray_tracing is True  # Discrete supports RT
    assert discrete_features.mesh_shaders is True
    assert discrete_features.bindless is True
    assert discrete_features.compute is True
    assert discrete_features.max_texture_size > 0

    # Integrated has fewer features
    integrated_adapter = NullAdapter(AdapterType.INTEGRATED)
    integrated_features = integrated_adapter.query_features()
    assert integrated_features.ray_tracing is False


def test_adapter_query_format_support():
    """Test format support query."""
    adapter = NullAdapter()
    format_support = adapter.query_format_support(Format.RGBA8_UNORM)

    assert format_support.renderable is True
    assert format_support.filterable is True
    assert format_support.blendable is True
    assert format_support.storage is True
    assert format_support.multisample is True


def test_device_create():
    """Test device creation with queues available."""
    adapter = NullAdapter()
    config = DeviceConfig(adapter=adapter, enable_debug=True, enable_validation=True)
    device = NullDevice.create(adapter, config)

    assert isinstance(device, Device)
    assert device is not None

    # Verify queues are available
    graphics_queue = device.get_queue(QueueType.GRAPHICS)
    assert graphics_queue is not None


def test_device_get_queue():
    """Test queue retrieval."""
    adapter = NullAdapter()
    config = DeviceConfig(adapter=adapter)
    device = NullDevice.create(adapter, config)

    graphics_queue = device.get_queue(QueueType.GRAPHICS)
    compute_queue = device.get_queue(QueueType.COMPUTE)
    transfer_queue = device.get_queue(QueueType.TRANSFER)

    assert graphics_queue is not None
    assert compute_queue is not None
    assert transfer_queue is not None

    # Same queue instance returned
    assert device.get_queue(QueueType.GRAPHICS) is graphics_queue


def test_device_create_buffer():
    """Test buffer creation."""
    from engine.platform.rhi import BufferDesc, BufferUsage, MemoryType

    adapter = NullAdapter()
    device = NullDevice.create(adapter, DeviceConfig(adapter=adapter))

    buffer_desc = BufferDesc(
        size=1024,
        usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,
        memory_type=MemoryType.DEFAULT
    )
    buffer = device.create_buffer(buffer_desc)

    assert buffer is not None
    assert buffer.is_valid()
    assert buffer.desc.size == 1024


def test_device_create_texture():
    """Test texture creation."""
    from engine.platform.rhi import TextureDesc, TextureType, TextureUsage, SampleCount

    adapter = NullAdapter()
    device = NullDevice.create(adapter, DeviceConfig(adapter=adapter))

    texture_desc = TextureDesc(
        type=TextureType.TEXTURE_2D,
        format=Format.RGBA8_UNORM,
        width=1024,
        height=768,
        usage=TextureUsage.RENDER_TARGET | TextureUsage.SHADER_RESOURCE,
        sample_count=SampleCount.X1
    )
    texture = device.create_texture(texture_desc)

    assert texture is not None
    assert texture.is_valid()
    assert texture.desc.width == 1024
    assert texture.desc.height == 768


def test_device_create_sampler():
    """Test sampler creation."""
    from engine.platform.rhi import SamplerDesc, FilterMode, AddressMode

    adapter = NullAdapter()
    device = NullDevice.create(adapter, DeviceConfig(adapter=adapter))

    sampler_desc = SamplerDesc(
        min_filter=FilterMode.LINEAR,
        mag_filter=FilterMode.LINEAR,
        address_u=AddressMode.WRAP,
        address_v=AddressMode.WRAP
    )
    sampler = device.create_sampler(sampler_desc)

    assert sampler is not None
    assert sampler.is_valid()


def test_device_create_pipeline():
    """Test pipeline creation."""
    from engine.platform.rhi import GraphicsPipelineDesc, ComputePipelineDesc, ShaderDesc, ShaderStage

    adapter = NullAdapter()
    device = NullDevice.create(adapter, DeviceConfig(adapter=adapter))

    # Graphics pipeline
    vs_desc = ShaderDesc(stage=ShaderStage.VERTEX, source=b"vs_code", entry_point="vs_main")
    ps_desc = ShaderDesc(stage=ShaderStage.PIXEL, source=b"ps_code", entry_point="ps_main")
    graphics_desc = GraphicsPipelineDesc(
        vertex_shader=vs_desc,
        pixel_shader=ps_desc
    )
    graphics_pipeline = device.create_graphics_pipeline(graphics_desc)

    assert graphics_pipeline is not None
    assert graphics_pipeline.is_valid()

    # Compute pipeline
    cs_desc = ShaderDesc(stage=ShaderStage.COMPUTE, source=b"cs_code")
    compute_desc = ComputePipelineDesc(compute_shader=cs_desc)
    compute_pipeline = device.create_compute_pipeline(compute_desc)

    assert compute_pipeline is not None
    assert compute_pipeline.is_valid()


def test_device_wait_idle():
    """Test device wait idle."""
    adapter = NullAdapter()
    device = NullDevice.create(adapter, DeviceConfig(adapter=adapter))

    # Should not hang
    device.wait_idle()


def test_device_shutdown():
    """Test device shutdown."""
    adapter = NullAdapter()
    device = NullDevice.create(adapter, DeviceConfig(adapter=adapter))

    device.shutdown()
    assert device._shutdown_called is True
