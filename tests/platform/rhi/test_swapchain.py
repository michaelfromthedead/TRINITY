"""Tests for RHI swapchain."""
import pytest
from engine.platform.rhi import (
    Swapchain, SwapchainDesc, PresentMode, ColorSpace, Format,
    NullSwapchain, NullAdapter, NullDevice, DeviceConfig
)


@pytest.fixture
def device():
    """Create test device."""
    adapter = NullAdapter()
    return NullDevice.create(adapter, DeviceConfig(adapter=adapter))


def test_swapchain_creation(device):
    """Test swapchain creation."""
    desc = SwapchainDesc(
        width=1920,
        height=1080,
        format=Format.RGBA8_UNORM,
        buffer_count=2,
        present_mode=PresentMode.VSYNC,
        color_space=ColorSpace.SRGB
    )

    swapchain = NullSwapchain.create(device, desc)

    assert swapchain is not None
    assert isinstance(swapchain, Swapchain)


def test_swapchain_current_texture(device):
    """Test getting current back buffer texture."""
    desc = SwapchainDesc(
        width=800,
        height=600,
        format=Format.RGBA8_UNORM,
        buffer_count=3
    )

    swapchain = NullSwapchain.create(device, desc)
    texture = swapchain.current_texture()

    assert texture is not None
    assert texture.is_valid()
    assert texture.desc.width == 800
    assert texture.desc.height == 600


def test_swapchain_current_index(device):
    """Test getting current back buffer index."""
    desc = SwapchainDesc(
        width=1024,
        height=768,
        format=Format.RGBA8_UNORM,
        buffer_count=2
    )

    swapchain = NullSwapchain.create(device, desc)
    index = swapchain.current_index()

    assert index >= 0
    assert index < desc.buffer_count


def test_swapchain_present(device):
    """Test swapchain present."""
    desc = SwapchainDesc(
        width=1920,
        height=1080,
        format=Format.RGBA8_UNORM,
        buffer_count=3
    )

    swapchain = NullSwapchain.create(device, desc)
    initial_index = swapchain.current_index()

    swapchain.present()
    new_index = swapchain.current_index()

    # Index should advance
    assert new_index == (initial_index + 1) % desc.buffer_count


def test_swapchain_present_wraps(device):
    """Test swapchain present wraps around."""
    desc = SwapchainDesc(
        width=800,
        height=600,
        format=Format.RGBA8_UNORM,
        buffer_count=2
    )

    swapchain = NullSwapchain.create(device, desc)

    # Present multiple times to wrap
    for _ in range(5):
        swapchain.present()

    index = swapchain.current_index()
    assert index < desc.buffer_count


def test_swapchain_resize(device):
    """Test swapchain resize."""
    desc = SwapchainDesc(
        width=800,
        height=600,
        format=Format.RGBA8_UNORM,
        buffer_count=2
    )

    swapchain = NullSwapchain.create(device, desc)
    old_texture = swapchain.current_texture()
    old_handle = old_texture.handle

    # Resize
    swapchain.resize(1920, 1080)

    # Get new texture
    new_texture = swapchain.current_texture()

    assert new_texture.desc.width == 1920
    assert new_texture.desc.height == 1080
    # Should have new texture after resize
    assert new_texture.handle != old_handle


def test_swapchain_resize_resets_index(device):
    """Test swapchain resize resets buffer index."""
    desc = SwapchainDesc(
        width=640,
        height=480,
        format=Format.RGBA8_UNORM,
        buffer_count=3
    )

    swapchain = NullSwapchain.create(device, desc)

    # Advance to index 2
    swapchain.present()
    swapchain.present()
    assert swapchain.current_index() == 2

    # Resize should reset to 0
    swapchain.resize(1024, 768)
    assert swapchain.current_index() == 0


def test_swapchain_triple_buffering(device):
    """Test triple buffering configuration."""
    desc = SwapchainDesc(
        width=1920,
        height=1080,
        format=Format.RGBA8_UNORM,
        buffer_count=3,
        present_mode=PresentMode.MAILBOX
    )

    swapchain = NullSwapchain.create(device, desc)

    # All three buffers should be valid
    indices = set()
    for _ in range(3):
        indices.add(swapchain.current_index())
        swapchain.present()

    assert len(indices) == 3


def test_swapchain_present_modes(device):
    """Test creating swapchains with different present modes."""
    # VSYNC mode
    desc_vsync = SwapchainDesc(
        width=800, height=600,
        format=Format.RGBA8_UNORM,
        buffer_count=2,
        present_mode=PresentMode.VSYNC
    )
    swapchain_vsync = NullSwapchain.create(device, desc_vsync)
    assert swapchain_vsync is not None

    # MAILBOX mode
    desc_mailbox = SwapchainDesc(
        width=800, height=600,
        format=Format.RGBA8_UNORM,
        buffer_count=3,
        present_mode=PresentMode.MAILBOX
    )
    swapchain_mailbox = NullSwapchain.create(device, desc_mailbox)
    assert swapchain_mailbox is not None


def test_swapchain_color_spaces(device):
    """Test creating swapchains with different color spaces."""
    # SRGB color space
    desc_srgb = SwapchainDesc(
        width=1920, height=1080,
        format=Format.RGBA8_UNORM,
        buffer_count=2,
        color_space=ColorSpace.SRGB
    )
    swapchain_srgb = NullSwapchain.create(device, desc_srgb)
    assert swapchain_srgb is not None

    # HDR10 color space
    desc_hdr = SwapchainDesc(
        width=1920, height=1080,
        format=Format.RGBA16_FLOAT,
        buffer_count=2,
        color_space=ColorSpace.HDR10
    )
    swapchain_hdr = NullSwapchain.create(device, desc_hdr)
    assert swapchain_hdr is not None


def test_swapchain_hdr_format(device):
    """Test HDR swapchain creation."""
    desc = SwapchainDesc(
        width=3840,
        height=2160,
        format=Format.RGBA16_FLOAT,
        buffer_count=2,
        present_mode=PresentMode.VSYNC,
        color_space=ColorSpace.HDR10
    )

    swapchain = NullSwapchain.create(device, desc)
    texture = swapchain.current_texture()

    assert texture.desc.format == Format.RGBA16_FLOAT
