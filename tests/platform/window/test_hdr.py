"""Tests for HDR support."""

import pytest

from engine.platform.window import DisplayHDR, HDRCapabilities, ColorSpace


class TestHDRSupport:
    """Tests for HDR support detection."""

    def test_hdr_not_supported_by_default(self):
        """Test that HDR is not supported by default in headless mode."""
        hdr = DisplayHDR()
        assert hdr.is_supported() is False

    def test_hdr_manager_creation(self):
        """Test creating HDR manager."""
        hdr = DisplayHDR()
        assert hdr is not None

    def test_hdr_simulation(self):
        """Test HDR manager with simulation."""
        hdr = DisplayHDR(simulate_hdr=True)
        caps = hdr.get_capabilities()
        assert caps.supported is True


class TestHDRCapabilities:
    """Tests for HDR capabilities."""

    def test_capabilities_without_hdr(self):
        """Test capabilities when HDR is not supported."""
        hdr = DisplayHDR(simulate_hdr=False)
        caps = hdr.get_capabilities()
        assert caps.supported is False
        assert caps.color_space == ColorSpace.SRGB
        assert caps.max_luminance == 100.0

    def test_capabilities_with_hdr(self):
        """Test capabilities when HDR is simulated."""
        hdr = DisplayHDR(simulate_hdr=True)
        caps = hdr.get_capabilities()
        assert caps.supported is True
        assert caps.color_space == ColorSpace.HDR10
        assert caps.min_luminance < caps.max_luminance
        assert caps.max_full_frame_luminance > 0

    def test_luminance_ranges(self):
        """Test that luminance values are in valid ranges."""
        hdr = DisplayHDR(simulate_hdr=True)
        caps = hdr.get_capabilities()
        assert caps.min_luminance >= 0.0
        assert caps.max_luminance > caps.min_luminance
        assert caps.max_full_frame_luminance <= caps.max_luminance


class TestColorSpace:
    """Tests for color space management."""

    def test_default_color_space(self):
        """Test default color space."""
        hdr = DisplayHDR(simulate_hdr=False)
        assert hdr.current_color_space == ColorSpace.SRGB

    def test_set_srgb_color_space(self):
        """Test setting sRGB color space."""
        hdr = DisplayHDR(simulate_hdr=True)
        result = hdr.set_color_space(ColorSpace.SRGB)
        assert result is True
        assert hdr.current_color_space == ColorSpace.SRGB

    def test_set_hdr10_color_space(self):
        """Test setting HDR10 color space."""
        hdr = DisplayHDR(simulate_hdr=True)
        result = hdr.set_color_space(ColorSpace.HDR10)
        assert result is True
        assert hdr.current_color_space == ColorSpace.HDR10

    def test_set_scrgb_color_space(self):
        """Test setting scRGB color space."""
        hdr = DisplayHDR(simulate_hdr=True)
        result = hdr.set_color_space(ColorSpace.SCRGB)
        assert result is True
        assert hdr.current_color_space == ColorSpace.SCRGB

    def test_set_pq_color_space(self):
        """Test setting PQ color space."""
        hdr = DisplayHDR(simulate_hdr=True)
        result = hdr.set_color_space(ColorSpace.PQ)
        assert result is True
        assert hdr.current_color_space == ColorSpace.PQ

    def test_cannot_set_hdr_without_support(self):
        """Test that HDR color spaces cannot be set without support."""
        hdr = DisplayHDR(simulate_hdr=False)
        result = hdr.set_color_space(ColorSpace.HDR10)
        assert result is False
        assert hdr.current_color_space == ColorSpace.SRGB

    def test_can_set_srgb_without_support(self):
        """Test that sRGB can always be set."""
        hdr = DisplayHDR(simulate_hdr=False)
        result = hdr.set_color_space(ColorSpace.SRGB)
        assert result is True


class TestHDRMetadata:
    """Tests for HDR metadata management."""

    def test_default_metadata(self):
        """Test default metadata is empty."""
        hdr = DisplayHDR()
        metadata = hdr.metadata
        assert len(metadata) == 0

    def test_set_max_content_light_level(self):
        """Test setting max content light level."""
        hdr = DisplayHDR()
        hdr.set_metadata(max_content_light_level=1000.0)
        metadata = hdr.metadata
        assert "max_cll" in metadata
        assert metadata["max_cll"] == 1000.0

    def test_set_max_frame_average_light_level(self):
        """Test setting max frame average light level."""
        hdr = DisplayHDR()
        hdr.set_metadata(max_frame_average_light_level=400.0)
        metadata = hdr.metadata
        assert "max_fall" in metadata
        assert metadata["max_fall"] == 400.0

    def test_set_mastering_luminance(self):
        """Test setting mastering display luminance."""
        hdr = DisplayHDR()
        hdr.set_metadata(
            min_mastering_luminance=0.0001,
            max_mastering_luminance=1000.0
        )
        metadata = hdr.metadata
        assert "min_mdl" in metadata
        assert "max_mdl" in metadata
        assert metadata["min_mdl"] == 0.0001
        assert metadata["max_mdl"] == 1000.0

    def test_set_all_metadata(self):
        """Test setting all metadata fields."""
        hdr = DisplayHDR()
        hdr.set_metadata(
            max_content_light_level=1000.0,
            max_frame_average_light_level=400.0,
            min_mastering_luminance=0.0001,
            max_mastering_luminance=1000.0
        )
        metadata = hdr.metadata
        assert len(metadata) == 4
        assert "max_cll" in metadata
        assert "max_fall" in metadata
        assert "min_mdl" in metadata
        assert "max_mdl" in metadata

    def test_metadata_immutability(self):
        """Test that metadata property returns a copy."""
        hdr = DisplayHDR()
        hdr.set_metadata(max_content_light_level=1000.0)
        metadata1 = hdr.metadata
        metadata1["max_cll"] = 2000.0
        metadata2 = hdr.metadata
        assert metadata2["max_cll"] == 1000.0  # Original value


class TestHDRCapabilitiesDataclass:
    """Tests for HDRCapabilities dataclass."""

    def test_create_capabilities(self):
        """Test creating HDR capabilities."""
        caps = HDRCapabilities(
            supported=True,
            min_luminance=0.0001,
            max_luminance=1000.0,
            max_full_frame_luminance=400.0,
            color_space=ColorSpace.HDR10
        )
        assert caps.supported is True
        assert caps.min_luminance == 0.0001
        assert caps.max_luminance == 1000.0
        assert caps.max_full_frame_luminance == 400.0
        assert caps.color_space == ColorSpace.HDR10


class TestColorSpaceEnum:
    """Tests for ColorSpace enum."""

    def test_color_space_workflow_with_hdr(self):
        """Test setting HDR color spaces on HDR vs non-HDR displays."""
        # HDR display - should accept HDR color spaces
        hdr_display = DisplayHDR(simulate_hdr=True)
        result = hdr_display.set_color_space(ColorSpace.HDR10)
        assert result is True
        assert hdr_display.current_color_space == ColorSpace.HDR10

        # Non-HDR display - should reject HDR color spaces
        non_hdr_display = DisplayHDR(simulate_hdr=False)
        result = non_hdr_display.set_color_space(ColorSpace.HDR10)
        assert result is False
        assert non_hdr_display.current_color_space == ColorSpace.SRGB

    def test_color_space_transitions(self):
        """Test transitioning between different color spaces."""
        hdr = DisplayHDR(simulate_hdr=True)

        # When simulate_hdr=True, starts with HDR10
        assert hdr.current_color_space == ColorSpace.HDR10

        # Transition to scRGB
        result = hdr.set_color_space(ColorSpace.SCRGB)
        assert result is True
        assert hdr.current_color_space == ColorSpace.SCRGB

        # Transition back to HDR10
        result = hdr.set_color_space(ColorSpace.HDR10)
        assert result is True
        assert hdr.current_color_space == ColorSpace.HDR10

        # Transition to scRGB
        result = hdr.set_color_space(ColorSpace.SCRGB)
        assert result is True
        assert hdr.current_color_space == ColorSpace.SCRGB

        # Back to sRGB
        result = hdr.set_color_space(ColorSpace.SRGB)
        assert result is True
        assert hdr.current_color_space == ColorSpace.SRGB
