"""
Tests for Denoiser System

Tests A-Trous wavelet spatial denoising for ray tracing:
- DenoiserQuality enum values
- DenoiserParams defaults and validation
- GBuffer validation
- Denoiser instantiation and lifecycle
- Iteration count per quality level
- Ping-pong buffer creation
- Spatial denoise dispatch
"""

import pytest
from unittest.mock import MagicMock, PropertyMock

from engine.rendering.postprocess.denoiser import (
    Denoiser,
    DenoiserParams,
    DenoiserQuality,
    GBuffer,
    create_default_params,
    create_quality_params,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_device():
    """Create a mock RHI device."""
    device = MagicMock()

    def create_texture_impl(desc):
        texture = MagicMock()
        texture.desc = desc
        texture.is_valid.return_value = True
        return texture

    device.create_texture.side_effect = create_texture_impl
    return device


@pytest.fixture
def mock_texture():
    """Create a mock texture with valid state."""
    texture = MagicMock()
    desc = MagicMock()
    desc.width = 1920
    desc.height = 1080
    type(texture).desc = PropertyMock(return_value=desc)
    texture.is_valid.return_value = True
    return texture


@pytest.fixture
def mock_output_texture():
    """Create a mock output texture matching input dimensions."""
    texture = MagicMock()
    desc = MagicMock()
    desc.width = 1920
    desc.height = 1080
    type(texture).desc = PropertyMock(return_value=desc)
    texture.is_valid.return_value = True
    return texture


@pytest.fixture
def mock_g_buffer():
    """Create a mock G-Buffer with valid textures."""
    depth = MagicMock()
    depth.is_valid.return_value = True

    normal = MagicMock()
    normal.is_valid.return_value = True

    albedo = MagicMock()
    albedo.is_valid.return_value = True

    return GBuffer(depth=depth, normal=normal, albedo=albedo)


# =============================================================================
# DenoiserQuality Tests
# =============================================================================


class TestDenoiserQuality:
    """Test DenoiserQuality enum."""

    def test_quality_low_value(self):
        """Test LOW quality has correct iteration count."""
        assert DenoiserQuality.LOW == 2

    def test_quality_medium_value(self):
        """Test MEDIUM quality has correct iteration count."""
        assert DenoiserQuality.MEDIUM == 3

    def test_quality_high_value(self):
        """Test HIGH quality has correct iteration count."""
        assert DenoiserQuality.HIGH == 4

    def test_quality_ultra_value(self):
        """Test ULTRA quality has correct iteration count."""
        assert DenoiserQuality.ULTRA == 5

    def test_quality_is_int_enum(self):
        """Test that quality values can be used as integers."""
        assert int(DenoiserQuality.LOW) == 2
        assert int(DenoiserQuality.MEDIUM) == 3
        assert int(DenoiserQuality.HIGH) == 4
        assert int(DenoiserQuality.ULTRA) == 5

    def test_quality_comparison(self):
        """Test quality level comparisons."""
        assert DenoiserQuality.LOW < DenoiserQuality.MEDIUM
        assert DenoiserQuality.MEDIUM < DenoiserQuality.HIGH
        assert DenoiserQuality.HIGH < DenoiserQuality.ULTRA

    def test_quality_iteration_arithmetic(self):
        """Test using quality value in arithmetic."""
        quality = DenoiserQuality.HIGH
        step_size = 1 << (quality - 1)  # 2^(quality-1)
        assert step_size == 8


# =============================================================================
# DenoiserParams Tests
# =============================================================================


class TestDenoiserParams:
    """Test DenoiserParams dataclass."""

    def test_params_defaults(self):
        """Test default parameter values."""
        params = DenoiserParams()

        assert params.quality == DenoiserQuality.MEDIUM
        assert params.sigma_color == 1.0
        assert params.sigma_depth == 1.0
        assert params.sigma_normal == 1.0

    def test_params_custom_quality(self):
        """Test creating params with custom quality."""
        params = DenoiserParams(quality=DenoiserQuality.ULTRA)

        assert params.quality == DenoiserQuality.ULTRA

    def test_params_custom_sigmas(self):
        """Test creating params with custom sigma values."""
        params = DenoiserParams(
            sigma_color=0.5,
            sigma_depth=2.0,
            sigma_normal=1.5,
        )

        assert params.sigma_color == 0.5
        assert params.sigma_depth == 2.0
        assert params.sigma_normal == 1.5

    def test_params_invalid_quality_type(self):
        """Test that invalid quality type raises TypeError."""
        with pytest.raises(TypeError, match="quality must be DenoiserQuality"):
            DenoiserParams(quality=3)

    def test_params_invalid_sigma_color(self):
        """Test that non-positive sigma_color raises ValueError."""
        with pytest.raises(ValueError, match="sigma_color must be positive"):
            DenoiserParams(sigma_color=0.0)

        with pytest.raises(ValueError, match="sigma_color must be positive"):
            DenoiserParams(sigma_color=-1.0)

    def test_params_invalid_sigma_depth(self):
        """Test that non-positive sigma_depth raises ValueError."""
        with pytest.raises(ValueError, match="sigma_depth must be positive"):
            DenoiserParams(sigma_depth=0.0)

    def test_params_invalid_sigma_normal(self):
        """Test that non-positive sigma_normal raises ValueError."""
        with pytest.raises(ValueError, match="sigma_normal must be positive"):
            DenoiserParams(sigma_normal=-0.5)

    def test_params_get_shader_params(self):
        """Test getting shader parameter tuple."""
        params = DenoiserParams(
            sigma_color=0.8,
            sigma_depth=1.2,
            sigma_normal=0.9,
        )

        shader_params = params.get_shader_params()

        assert shader_params == (0.8, 1.2, 0.9)
        assert len(shader_params) == 3


# =============================================================================
# GBuffer Tests
# =============================================================================


class TestGBuffer:
    """Test GBuffer dataclass."""

    def test_gbuffer_creation(self, mock_g_buffer):
        """Test G-Buffer creation with textures."""
        assert mock_g_buffer.depth is not None
        assert mock_g_buffer.normal is not None
        assert mock_g_buffer.albedo is not None

    def test_gbuffer_is_valid(self, mock_g_buffer):
        """Test G-Buffer validity check."""
        assert mock_g_buffer.is_valid()

    def test_gbuffer_invalid_without_depth(self):
        """Test G-Buffer is invalid without depth."""
        normal = MagicMock()
        normal.is_valid.return_value = True

        g_buffer = GBuffer(depth=None, normal=normal)
        assert not g_buffer.is_valid()

    def test_gbuffer_invalid_without_normal(self):
        """Test G-Buffer is invalid without normal."""
        depth = MagicMock()
        depth.is_valid.return_value = True

        g_buffer = GBuffer(depth=depth, normal=None)
        assert not g_buffer.is_valid()

    def test_gbuffer_invalid_depth_texture(self):
        """Test G-Buffer is invalid when depth texture is invalid."""
        depth = MagicMock()
        depth.is_valid.return_value = False

        normal = MagicMock()
        normal.is_valid.return_value = True

        g_buffer = GBuffer(depth=depth, normal=normal)
        assert not g_buffer.is_valid()

    def test_gbuffer_has_albedo_true(self, mock_g_buffer):
        """Test has_albedo returns True when albedo present."""
        assert mock_g_buffer.has_albedo()

    def test_gbuffer_has_albedo_false_none(self):
        """Test has_albedo returns False when albedo is None."""
        depth = MagicMock()
        depth.is_valid.return_value = True
        normal = MagicMock()
        normal.is_valid.return_value = True

        g_buffer = GBuffer(depth=depth, normal=normal, albedo=None)
        assert not g_buffer.has_albedo()

    def test_gbuffer_has_albedo_false_invalid(self):
        """Test has_albedo returns False when albedo is invalid."""
        depth = MagicMock()
        depth.is_valid.return_value = True
        normal = MagicMock()
        normal.is_valid.return_value = True
        albedo = MagicMock()
        albedo.is_valid.return_value = False

        g_buffer = GBuffer(depth=depth, normal=normal, albedo=albedo)
        assert not g_buffer.has_albedo()


# =============================================================================
# Denoiser Instantiation Tests
# =============================================================================


class TestDenoiserInstantiation:
    """Test Denoiser class instantiation."""

    def test_denoiser_creation(self, mock_device):
        """Test denoiser creation with device."""
        denoiser = Denoiser(mock_device)

        assert denoiser is not None
        assert denoiser.device is mock_device

    def test_denoiser_not_initialized(self, mock_device):
        """Test denoiser is not initialized on creation."""
        denoiser = Denoiser(mock_device)

        assert not denoiser.is_initialized

    def test_denoiser_device_property(self, mock_device):
        """Test device property returns correct device."""
        denoiser = Denoiser(mock_device)

        assert denoiser.device is mock_device


# =============================================================================
# Iteration Count Tests
# =============================================================================


class TestIterationCount:
    """Test iteration count per quality level."""

    def test_iteration_count_low(self, mock_device):
        """Test LOW quality returns 2 iterations."""
        denoiser = Denoiser(mock_device)
        assert denoiser.get_iteration_count(DenoiserQuality.LOW) == 2

    def test_iteration_count_medium(self, mock_device):
        """Test MEDIUM quality returns 3 iterations."""
        denoiser = Denoiser(mock_device)
        assert denoiser.get_iteration_count(DenoiserQuality.MEDIUM) == 3

    def test_iteration_count_high(self, mock_device):
        """Test HIGH quality returns 4 iterations."""
        denoiser = Denoiser(mock_device)
        assert denoiser.get_iteration_count(DenoiserQuality.HIGH) == 4

    def test_iteration_count_ultra(self, mock_device):
        """Test ULTRA quality returns 5 iterations."""
        denoiser = Denoiser(mock_device)
        assert denoiser.get_iteration_count(DenoiserQuality.ULTRA) == 5


# =============================================================================
# Ping-Pong Buffer Tests
# =============================================================================


class TestPingPongBuffers:
    """Test ping-pong buffer creation."""

    def test_create_ping_pong_buffers(self, mock_device):
        """Test creating ping-pong buffers."""
        denoiser = Denoiser(mock_device)
        ping, pong = denoiser.create_ping_pong_buffers(1920, 1080)

        assert ping is not None
        assert pong is not None
        assert mock_device.create_texture.call_count == 2

    def test_ping_pong_buffers_dimensions(self, mock_device):
        """Test ping-pong buffers have correct dimensions."""
        denoiser = Denoiser(mock_device)
        ping, pong = denoiser.create_ping_pong_buffers(1920, 1080)

        assert ping.desc.width == 1920
        assert ping.desc.height == 1080
        assert pong.desc.width == 1920
        assert pong.desc.height == 1080

    def test_ping_pong_buffers_reuse(self, mock_device):
        """Test ping-pong buffers are reused when dimensions match."""
        denoiser = Denoiser(mock_device)

        # First creation
        ping1, pong1 = denoiser.create_ping_pong_buffers(1920, 1080)
        call_count_1 = mock_device.create_texture.call_count

        # Second call with same dimensions should reuse
        ping2, pong2 = denoiser.create_ping_pong_buffers(1920, 1080)
        call_count_2 = mock_device.create_texture.call_count

        assert call_count_1 == call_count_2
        assert ping1 is ping2
        assert pong1 is pong2

    def test_ping_pong_buffers_recreate_on_resize(self, mock_device):
        """Test ping-pong buffers are recreated when dimensions change."""
        denoiser = Denoiser(mock_device)

        # First creation
        ping1, pong1 = denoiser.create_ping_pong_buffers(1920, 1080)
        call_count_1 = mock_device.create_texture.call_count

        # Second call with different dimensions should create new buffers
        ping2, pong2 = denoiser.create_ping_pong_buffers(2560, 1440)
        call_count_2 = mock_device.create_texture.call_count

        assert call_count_2 == call_count_1 + 2
        assert ping1 is not ping2
        assert pong1 is not pong2

    def test_ping_pong_invalid_width(self, mock_device):
        """Test invalid width raises ValueError."""
        denoiser = Denoiser(mock_device)

        with pytest.raises(ValueError, match="width must be positive"):
            denoiser.create_ping_pong_buffers(0, 1080)

        with pytest.raises(ValueError, match="width must be positive"):
            denoiser.create_ping_pong_buffers(-100, 1080)

    def test_ping_pong_invalid_height(self, mock_device):
        """Test invalid height raises ValueError."""
        denoiser = Denoiser(mock_device)

        with pytest.raises(ValueError, match="height must be positive"):
            denoiser.create_ping_pong_buffers(1920, 0)

        with pytest.raises(ValueError, match="height must be positive"):
            denoiser.create_ping_pong_buffers(1920, -50)

    def test_initialized_after_buffer_creation(self, mock_device):
        """Test denoiser is marked initialized after buffer creation."""
        denoiser = Denoiser(mock_device)

        assert not denoiser.is_initialized
        denoiser.create_ping_pong_buffers(1920, 1080)
        assert denoiser.is_initialized


# =============================================================================
# Spatial Denoise Tests
# =============================================================================


class TestSpatialDenoise:
    """Test spatial denoise dispatch."""

    def test_spatial_denoise_default_params(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test spatial denoise with default params."""
        denoiser = Denoiser(mock_device)

        # Should not raise
        denoiser.spatial_denoise(
            noisy_input=mock_texture,
            g_buffer=mock_g_buffer,
            output=mock_output_texture,
        )

    def test_spatial_denoise_custom_params(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test spatial denoise with custom params."""
        denoiser = Denoiser(mock_device)
        params = DenoiserParams(
            quality=DenoiserQuality.ULTRA,
            sigma_color=0.5,
        )

        # Should not raise
        denoiser.spatial_denoise(
            noisy_input=mock_texture,
            g_buffer=mock_g_buffer,
            output=mock_output_texture,
            params=params,
        )

    def test_spatial_denoise_creates_buffers(
        self, mock_device, mock_texture, mock_output_texture, mock_g_buffer
    ):
        """Test spatial denoise creates ping-pong buffers."""
        denoiser = Denoiser(mock_device)

        assert not denoiser.is_initialized

        denoiser.spatial_denoise(
            noisy_input=mock_texture,
            g_buffer=mock_g_buffer,
            output=mock_output_texture,
        )

        assert denoiser.is_initialized

    def test_spatial_denoise_invalid_input(
        self, mock_device, mock_output_texture, mock_g_buffer
    ):
        """Test spatial denoise with invalid input texture."""
        denoiser = Denoiser(mock_device)

        with pytest.raises(ValueError, match="noisy_input texture is invalid"):
            denoiser.spatial_denoise(
                noisy_input=None,
                g_buffer=mock_g_buffer,
                output=mock_output_texture,
            )

    def test_spatial_denoise_invalid_output(
        self, mock_device, mock_texture, mock_g_buffer
    ):
        """Test spatial denoise with invalid output texture."""
        denoiser = Denoiser(mock_device)

        with pytest.raises(ValueError, match="output texture is invalid"):
            denoiser.spatial_denoise(
                noisy_input=mock_texture,
                g_buffer=mock_g_buffer,
                output=None,
            )

    def test_spatial_denoise_invalid_gbuffer(
        self, mock_device, mock_texture, mock_output_texture
    ):
        """Test spatial denoise with invalid G-Buffer."""
        denoiser = Denoiser(mock_device)

        invalid_gbuffer = GBuffer(depth=None, normal=None)

        with pytest.raises(ValueError, match="g_buffer is invalid"):
            denoiser.spatial_denoise(
                noisy_input=mock_texture,
                g_buffer=invalid_gbuffer,
                output=mock_output_texture,
            )

    def test_spatial_denoise_dimension_mismatch(
        self, mock_device, mock_texture, mock_g_buffer
    ):
        """Test spatial denoise with mismatched dimensions."""
        denoiser = Denoiser(mock_device)

        # Create output with different dimensions
        output = MagicMock()
        output_desc = MagicMock()
        output_desc.width = 1280
        output_desc.height = 720
        type(output).desc = PropertyMock(return_value=output_desc)
        output.is_valid.return_value = True

        with pytest.raises(ValueError, match="Output dimensions.*do not match"):
            denoiser.spatial_denoise(
                noisy_input=mock_texture,
                g_buffer=mock_g_buffer,
                output=output,
            )


# =============================================================================
# Denoiser Lifecycle Tests
# =============================================================================


class TestDenoiserLifecycle:
    """Test denoiser resource lifecycle."""

    def test_destroy_releases_buffers(self, mock_device):
        """Test destroy releases ping-pong buffers."""
        denoiser = Denoiser(mock_device)
        ping, pong = denoiser.create_ping_pong_buffers(1920, 1080)

        denoiser.destroy()

        assert not denoiser.is_initialized
        ping.destroy.assert_called_once()
        pong.destroy.assert_called_once()

    def test_destroy_without_buffers(self, mock_device):
        """Test destroy is safe without buffers."""
        denoiser = Denoiser(mock_device)

        # Should not raise
        denoiser.destroy()
        assert not denoiser.is_initialized

    def test_destroy_idempotent(self, mock_device):
        """Test destroy can be called multiple times."""
        denoiser = Denoiser(mock_device)
        denoiser.create_ping_pong_buffers(1920, 1080)

        denoiser.destroy()
        denoiser.destroy()  # Should not raise

        assert not denoiser.is_initialized


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_default_params(self):
        """Test create_default_params returns correct defaults."""
        params = create_default_params()

        assert params.quality == DenoiserQuality.MEDIUM
        assert params.sigma_color == 1.0
        assert params.sigma_depth == 1.0
        assert params.sigma_normal == 1.0

    def test_create_quality_params_low(self):
        """Test create_quality_params for LOW quality."""
        params = create_quality_params(DenoiserQuality.LOW)

        assert params.quality == DenoiserQuality.LOW
        assert params.sigma_color == 1.2

    def test_create_quality_params_medium(self):
        """Test create_quality_params for MEDIUM quality."""
        params = create_quality_params(DenoiserQuality.MEDIUM)

        assert params.quality == DenoiserQuality.MEDIUM
        assert params.sigma_color == 1.0

    def test_create_quality_params_high(self):
        """Test create_quality_params for HIGH quality."""
        params = create_quality_params(DenoiserQuality.HIGH)

        assert params.quality == DenoiserQuality.HIGH
        assert params.sigma_color == 0.9

    def test_create_quality_params_ultra(self):
        """Test create_quality_params for ULTRA quality."""
        params = create_quality_params(DenoiserQuality.ULTRA)

        assert params.quality == DenoiserQuality.ULTRA
        assert params.sigma_color == 0.8
