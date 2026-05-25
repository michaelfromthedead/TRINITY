"""Tests for RHI resources."""
import pytest
from engine.platform.rhi import (
    Buffer, BufferDesc, BufferUsage, MemoryType,
    Texture, TextureDesc, TextureType, TextureUsage, Format, SampleCount,
    Sampler, SamplerDesc, FilterMode, AddressMode, CompareOp,
    NullAdapter, NullDevice, DeviceConfig
)


@pytest.fixture
def device():
    """Create test device."""
    adapter = NullAdapter()
    return NullDevice.create(adapter, DeviceConfig(adapter=adapter))


def test_buffer_creation(device):
    """Test buffer creation and properties."""
    desc = BufferDesc(
        size=4096,
        usage=BufferUsage.VERTEX | BufferUsage.INDEX,
        memory_type=MemoryType.DEFAULT,
        stride=16
    )
    buffer = device.create_buffer(desc)

    assert buffer.is_valid()
    assert buffer.handle > 0
    assert buffer.desc.size == 4096
    assert buffer.desc.stride == 16
    assert BufferUsage.VERTEX in buffer.desc.usage
    assert BufferUsage.INDEX in buffer.desc.usage


def test_buffer_destroy(device):
    """Test buffer destruction."""
    desc = BufferDesc(size=1024, usage=BufferUsage.CONSTANT)
    buffer = device.create_buffer(desc)

    assert buffer.is_valid()
    buffer.destroy()
    assert not buffer.is_valid()


def test_buffer_unique_handles(device):
    """Test that buffers get unique handles."""
    desc = BufferDesc(size=1024, usage=BufferUsage.STORAGE)
    buffer1 = device.create_buffer(desc)
    buffer2 = device.create_buffer(desc)

    assert buffer1.handle != buffer2.handle


def test_texture_creation(device):
    """Test texture creation and properties."""
    desc = TextureDesc(
        type=TextureType.TEXTURE_2D,
        format=Format.RGBA8_UNORM,
        width=512,
        height=512,
        mip_levels=4,
        usage=TextureUsage.SHADER_RESOURCE | TextureUsage.RENDER_TARGET,
        sample_count=SampleCount.X1
    )
    texture = device.create_texture(desc)

    assert texture.is_valid()
    assert texture.handle > 0
    assert texture.desc.width == 512
    assert texture.desc.height == 512
    assert texture.desc.mip_levels == 4
    assert texture.desc.format == Format.RGBA8_UNORM


def test_texture_3d_creation(device):
    """Test 3D texture creation."""
    desc = TextureDesc(
        type=TextureType.TEXTURE_3D,
        format=Format.RGBA16_FLOAT,
        width=128,
        height=128,
        depth=128,
        usage=TextureUsage.UNORDERED_ACCESS
    )
    texture = device.create_texture(desc)

    assert texture.is_valid()
    assert texture.desc.depth == 128


def test_texture_cube_creation(device):
    """Test cube texture creation."""
    desc = TextureDesc(
        type=TextureType.TEXTURE_CUBE,
        format=Format.RGBA8_UNORM,
        width=256,
        height=256,
        usage=TextureUsage.SHADER_RESOURCE
    )
    texture = device.create_texture(desc)

    assert texture.is_valid()


def test_texture_destroy(device):
    """Test texture destruction."""
    desc = TextureDesc(
        type=TextureType.TEXTURE_2D,
        format=Format.R32_FLOAT,
        width=64,
        height=64
    )
    texture = device.create_texture(desc)

    assert texture.is_valid()
    texture.destroy()
    assert not texture.is_valid()


def test_sampler_creation(device):
    """Test sampler creation and properties."""
    desc = SamplerDesc(
        min_filter=FilterMode.LINEAR,
        mag_filter=FilterMode.LINEAR,
        mip_filter=FilterMode.NEAREST,
        address_u=AddressMode.CLAMP,
        address_v=AddressMode.WRAP,
        address_w=AddressMode.MIRROR,
        max_anisotropy=16,
        compare_op=CompareOp.LESS,
        min_lod=0.0,
        max_lod=10.0
    )
    sampler = device.create_sampler(desc)

    assert sampler.is_valid()
    assert sampler.handle > 0
    assert sampler.desc.min_filter == FilterMode.LINEAR
    assert sampler.desc.address_u == AddressMode.CLAMP
    assert sampler.desc.max_anisotropy == 16


def test_sampler_destroy(device):
    """Test sampler destruction."""
    desc = SamplerDesc()
    sampler = device.create_sampler(desc)

    assert sampler.is_valid()
    sampler.destroy()
    assert not sampler.is_valid()


def test_buffer_usage_flags(device):
    """Test creating buffer with specific usage flags."""
    # Create buffer with VERTEX usage
    desc = BufferDesc(size=1024, usage=BufferUsage.VERTEX)
    buffer = device.create_buffer(desc)
    assert BufferUsage.VERTEX in buffer.desc.usage

    # Create buffer with combined usage
    desc2 = BufferDesc(size=2048, usage=BufferUsage.INDEX | BufferUsage.COPY_DST)
    buffer2 = device.create_buffer(desc2)
    assert BufferUsage.INDEX in buffer2.desc.usage
    assert BufferUsage.COPY_DST in buffer2.desc.usage
    assert BufferUsage.CONSTANT not in buffer2.desc.usage


def test_texture_usage_flags(device):
    """Test creating texture with usage flags stores them correctly."""
    usage = TextureUsage.RENDER_TARGET | TextureUsage.SHADER_RESOURCE
    desc = TextureDesc(
        type=TextureType.TEXTURE_2D,
        format=Format.RGBA8_UNORM,
        width=256, height=256,
        usage=usage
    )
    texture = device.create_texture(desc)

    assert TextureUsage.RENDER_TARGET in texture.desc.usage
    assert TextureUsage.SHADER_RESOURCE in texture.desc.usage
    assert TextureUsage.DEPTH_STENCIL not in texture.desc.usage


def test_format_creation_with_different_formats(device):
    """Test creating textures with different formats."""
    # RGBA8
    desc1 = TextureDesc(
        type=TextureType.TEXTURE_2D,
        format=Format.RGBA8_UNORM,
        width=128, height=128,
        usage=TextureUsage.SHADER_RESOURCE
    )
    tex1 = device.create_texture(desc1)
    assert tex1.desc.format == Format.RGBA8_UNORM

    # RGBA16 Float
    desc2 = TextureDesc(
        type=TextureType.TEXTURE_2D,
        format=Format.RGBA16_FLOAT,
        width=128, height=128,
        usage=TextureUsage.SHADER_RESOURCE
    )
    tex2 = device.create_texture(desc2)
    assert tex2.desc.format == Format.RGBA16_FLOAT
    assert tex2.desc.format != tex1.desc.format


def test_sample_count_in_texture_desc(device):
    """Test creating textures with different sample counts."""
    # X1 sample
    desc1 = TextureDesc(
        type=TextureType.TEXTURE_2D,
        format=Format.RGBA8_UNORM,
        width=256, height=256,
        usage=TextureUsage.RENDER_TARGET,
        sample_count=SampleCount.X1
    )
    tex1 = device.create_texture(desc1)
    assert tex1.desc.sample_count == SampleCount.X1
    assert tex1.desc.sample_count.value == 1

    # X4 MSAA
    desc4 = TextureDesc(
        type=TextureType.TEXTURE_2D,
        format=Format.RGBA8_UNORM,
        width=256, height=256,
        usage=TextureUsage.RENDER_TARGET,
        sample_count=SampleCount.X4
    )
    tex4 = device.create_texture(desc4)
    assert tex4.desc.sample_count == SampleCount.X4
    assert tex4.desc.sample_count.value == 4
