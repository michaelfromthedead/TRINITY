"""Tests for XR foveated rendering module."""

import pytest
import math

from engine.xr.rendering.foveated import (
    FoveationType,
    FoveationRegion,
    ShadingRate,
    FoveationRegionConfig,
    GazePoint,
    FoveationConfig,
    FoveationMetrics,
    FoveatedRenderer,
    FixedFoveatedRenderer,
    DynamicFoveatedRenderer,
    ContrastAdaptiveFoveatedRenderer,
    create_foveated_renderer,
)
from engine.xr.utils.shading import shading_rate_to_int, get_rate_multiplier


class TestFoveationConfig:
    """Tests for FoveationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FoveationConfig()

        assert config.type == FoveationType.FIXED
        assert config.enabled is True
        assert config.fovea_radius == pytest.approx(5.0)
        assert config.parafoveal_radius == pytest.approx(20.0)
        assert config.peripheral_radius == pytest.approx(55.0)
        assert config.fovea_rate == ShadingRate.FULL
        assert config.parafoveal_rate == ShadingRate.HALF
        assert config.peripheral_rate == ShadingRate.QUARTER

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FoveationConfig(
            type=FoveationType.DYNAMIC,
            fovea_radius=3.0,
            parafoveal_radius=15.0,
            gaze_smoothing=0.5
        )

        assert config.type == FoveationType.DYNAMIC
        assert config.fovea_radius == pytest.approx(3.0)
        assert config.gaze_smoothing == pytest.approx(0.5)


class TestGazePoint:
    """Tests for GazePoint dataclass."""

    def test_default_gaze(self):
        """Test default gaze point at center."""
        gaze = GazePoint()

        assert gaze.x == pytest.approx(0.0)
        assert gaze.y == pytest.approx(0.0)
        assert gaze.confidence == pytest.approx(1.0)

    def test_custom_gaze(self):
        """Test custom gaze point."""
        gaze = GazePoint(x=0.5, y=-0.3, confidence=0.9)

        assert gaze.x == pytest.approx(0.5)
        assert gaze.y == pytest.approx(-0.3)
        assert gaze.confidence == pytest.approx(0.9)


class TestFixedFoveatedRenderer:
    """Tests for FixedFoveatedRenderer."""

    def test_creation_default_config(self):
        """Test renderer creation with default config."""
        renderer = FixedFoveatedRenderer()

        assert renderer.config.type == FoveationType.FIXED
        assert renderer.config.enabled is True

    def test_creation_custom_config(self):
        """Test renderer creation with custom config."""
        config = FoveationConfig(fovea_radius=4.0)
        renderer = FixedFoveatedRenderer(config)

        assert renderer.config.fovea_radius == pytest.approx(4.0)

    def test_configure(self):
        """Test configuration update."""
        renderer = FixedFoveatedRenderer()

        new_config = FoveationConfig(fovea_radius=6.0)
        renderer.configure(new_config)

        # Should force FIXED type
        assert renderer.config.type == FoveationType.FIXED
        assert renderer.config.fovea_radius == pytest.approx(6.0)

    def test_update_gaze_ignored(self):
        """Test that gaze updates are ignored in fixed mode."""
        renderer = FixedFoveatedRenderer()

        # This should not affect anything in fixed mode
        gaze = GazePoint(x=0.8, y=0.8)
        renderer.update_gaze(gaze, gaze)

        # Region at center should still be FOVEA
        region = renderer.get_region_at_point(0.0, 0.0, 0)
        assert region == FoveationRegion.FOVEA

    def test_get_region_at_center(self):
        """Test region detection at screen center."""
        renderer = FixedFoveatedRenderer()

        region = renderer.get_region_at_point(0.0, 0.0, 0)

        assert region == FoveationRegion.FOVEA

    def test_get_region_at_periphery(self):
        """Test region detection at screen edge."""
        renderer = FixedFoveatedRenderer()

        # Far corner should be peripheral
        region = renderer.get_region_at_point(0.9, 0.9, 0)

        assert region == FoveationRegion.PERIPHERAL

    def test_get_region_at_parafoveal(self):
        """Test region detection in parafoveal zone."""
        renderer = FixedFoveatedRenderer()

        # Mid-distance from center
        region = renderer.get_region_at_point(0.3, 0.0, 0)

        assert region == FoveationRegion.PARAFOVEAL

    def test_get_shading_rate_image(self):
        """Test shading rate image generation."""
        renderer = FixedFoveatedRenderer()

        # Small test image
        rates = renderer.get_shading_rate_image(0, 8, 8)

        assert len(rates) == 64
        # All values should be valid rates (0-6)
        assert all(0 <= r <= 6 for r in rates)

    def test_get_shading_rate_image_center_is_full(self):
        """Test that center of shading rate image is full quality."""
        renderer = FixedFoveatedRenderer()

        rates = renderer.get_shading_rate_image(0, 16, 16)

        # Center pixel (approximately)
        center_idx = 8 * 16 + 8
        # FULL rate is 0
        assert rates[center_idx] == 0

    def test_get_metrics(self):
        """Test metrics reporting."""
        renderer = FixedFoveatedRenderer()
        renderer.begin_frame()
        renderer.end_frame()

        metrics = renderer.get_metrics()

        assert isinstance(metrics, FoveationMetrics)
        assert metrics.pixel_savings >= 0.0
        assert metrics.bandwidth_savings >= 0.0
        assert metrics.regions_active == 3

    def test_frame_lifecycle(self):
        """Test frame begin/end lifecycle."""
        renderer = FixedFoveatedRenderer()

        # Should not raise
        renderer.begin_frame()
        renderer.end_frame()


class TestDynamicFoveatedRenderer:
    """Tests for DynamicFoveatedRenderer."""

    def test_creation(self):
        """Test dynamic renderer creation."""
        renderer = DynamicFoveatedRenderer()

        assert renderer.config.type == FoveationType.DYNAMIC

    def test_gaze_update(self):
        """Test gaze point update."""
        renderer = DynamicFoveatedRenderer()

        left_gaze = GazePoint(x=0.5, y=0.3, confidence=1.0, timestamp_ns=1000)
        right_gaze = GazePoint(x=0.5, y=0.3, confidence=1.0, timestamp_ns=1000)

        renderer.update_gaze(left_gaze, right_gaze)

        # Region at new gaze point should be FOVEA
        region = renderer.get_region_at_point(0.5, 0.3, 0)
        # With smoothing, might not be exactly at gaze point yet
        # but should be FOVEA or PARAFOVEAL near the gaze
        assert region in (FoveationRegion.FOVEA, FoveationRegion.PARAFOVEAL)

    def test_gaze_smoothing(self):
        """Test gaze smoothing reduces jitter."""
        config = FoveationConfig(
            type=FoveationType.DYNAMIC,
            gaze_smoothing=0.8
        )
        renderer = DynamicFoveatedRenderer(config)

        # Initial gaze at center
        renderer.update_gaze(GazePoint(x=0.0, y=0.0), GazePoint(x=0.0, y=0.0))

        # Jump to far position
        renderer.update_gaze(GazePoint(x=1.0, y=0.0), GazePoint(x=1.0, y=0.0))

        # Smoothed position should not be at 1.0 yet
        metrics = renderer.get_metrics()
        assert metrics.current_gaze is not None
        assert metrics.current_gaze.x < 1.0

    def test_low_confidence_gaze_ignored(self):
        """Test that low confidence gaze updates are filtered."""
        config = FoveationConfig(
            type=FoveationType.DYNAMIC,
            confidence_threshold=0.5,
            gaze_smoothing=0.0  # No smoothing for clearer test
        )
        renderer = DynamicFoveatedRenderer(config)

        # Set initial high-confidence gaze
        renderer.update_gaze(
            GazePoint(x=0.0, y=0.0, confidence=1.0),
            GazePoint(x=0.0, y=0.0, confidence=1.0)
        )

        # Try to update with low confidence
        renderer.update_gaze(
            GazePoint(x=1.0, y=1.0, confidence=0.3),
            GazePoint(x=1.0, y=1.0, confidence=0.3)
        )

        # Gaze should not have moved to 1.0
        metrics = renderer.get_metrics()
        # With low confidence, original position should be maintained
        assert metrics.current_gaze.x < 0.5

    def test_shading_rate_follows_gaze(self):
        """Test that shading rate image centers on gaze."""
        renderer = DynamicFoveatedRenderer()

        # Set gaze to top-right
        renderer.update_gaze(
            GazePoint(x=0.8, y=0.8, confidence=1.0),
            GazePoint(x=0.8, y=0.8, confidence=1.0)
        )

        rates = renderer.get_shading_rate_image(0, 16, 16)

        # Top-right area should have more FULL rate pixels
        # Check approximate top-right quadrant
        top_right_full = 0
        for y in range(12, 16):
            for x in range(12, 16):
                if rates[y * 16 + x] == 0:  # FULL rate
                    top_right_full += 1

        # Should have some FULL rate in gaze area
        assert top_right_full > 0


class TestContrastAdaptiveFoveatedRenderer:
    """Tests for ContrastAdaptiveFoveatedRenderer."""

    def test_creation(self):
        """Test contrast-adaptive renderer creation."""
        renderer = ContrastAdaptiveFoveatedRenderer()

        assert renderer.config.type == FoveationType.CONTRAST_ADAPTIVE

    def test_contrast_callback(self):
        """Test contrast callback integration."""
        def high_contrast(x, y):
            return 1.0

        renderer = ContrastAdaptiveFoveatedRenderer(contrast_callback=high_contrast)
        rates = renderer.get_shading_rate_image(0, 8, 8)

        # With high contrast everywhere, should have more full-quality pixels
        full_count = sum(1 for r in rates if r == 0)
        assert full_count > 0

    def test_contrast_map_update(self):
        """Test contrast map update."""
        renderer = ContrastAdaptiveFoveatedRenderer()

        # Create contrast map with high contrast in center
        width, height = 8, 8
        contrast_map = [0.1] * (width * height)  # Low contrast base
        contrast_map[4 * width + 4] = 1.0  # High contrast at center

        renderer.update_contrast_map(contrast_map, width, height)

        # Should be able to get shading rates without error
        rates = renderer.get_shading_rate_image(0, width, height)
        assert len(rates) == width * height


class TestFoveatedRendererFactory:
    """Tests for create_foveated_renderer factory function."""

    def test_create_default(self):
        """Test default renderer creation."""
        renderer = create_foveated_renderer()

        assert isinstance(renderer, FixedFoveatedRenderer)

    def test_create_fixed(self):
        """Test Fixed renderer creation."""
        config = FoveationConfig(type=FoveationType.FIXED)
        renderer = create_foveated_renderer(config)

        assert isinstance(renderer, FixedFoveatedRenderer)

    def test_create_dynamic(self):
        """Test Dynamic renderer creation."""
        config = FoveationConfig(type=FoveationType.DYNAMIC)
        renderer = create_foveated_renderer(config)

        assert isinstance(renderer, DynamicFoveatedRenderer)

    def test_create_contrast_adaptive(self):
        """Test Contrast-Adaptive renderer creation."""
        config = FoveationConfig(type=FoveationType.CONTRAST_ADAPTIVE)
        renderer = create_foveated_renderer(config)

        assert isinstance(renderer, ContrastAdaptiveFoveatedRenderer)

    def test_create_disabled(self):
        """Test disabled foveation."""
        config = FoveationConfig(enabled=False)
        renderer = create_foveated_renderer(config)

        assert not renderer.config.enabled

    def test_create_none_type(self):
        """Test NONE foveation type."""
        config = FoveationConfig(type=FoveationType.NONE)
        renderer = create_foveated_renderer(config)

        assert not renderer.config.enabled


class TestShadingRates:
    """Tests for shading rate calculations."""

    def test_shading_rate_multipliers(self):
        """Test that shading rates have expected pixel coverage."""
        renderer = FixedFoveatedRenderer()

        # Verify rate multipliers
        assert get_rate_multiplier(ShadingRate.FULL) == pytest.approx(1.0)
        assert get_rate_multiplier(ShadingRate.HALF) == pytest.approx(0.25)
        assert get_rate_multiplier(ShadingRate.QUARTER) == pytest.approx(0.0625)

    def test_shading_rate_to_int(self):
        """Test shading rate enum to integer conversion."""
        assert shading_rate_to_int(ShadingRate.FULL) == 0
        assert shading_rate_to_int(ShadingRate.HALF) == 3
        assert shading_rate_to_int(ShadingRate.QUARTER) == 6


class TestFoveationMetrics:
    """Tests for foveation metrics."""

    def test_metrics_pixel_savings(self):
        """Test pixel savings calculation."""
        config = FoveationConfig(
            fovea_rate=ShadingRate.FULL,
            parafoveal_rate=ShadingRate.HALF,
            peripheral_rate=ShadingRate.QUARTER
        )
        renderer = FixedFoveatedRenderer(config)
        renderer.begin_frame()
        renderer.end_frame()

        metrics = renderer.get_metrics()

        # Should have significant pixel savings
        assert metrics.pixel_savings > 0.0
        assert metrics.pixel_savings < 100.0  # Not 100% savings

    def test_metrics_bandwidth_savings(self):
        """Test bandwidth savings calculation."""
        renderer = FixedFoveatedRenderer()
        renderer.begin_frame()
        renderer.end_frame()

        metrics = renderer.get_metrics()

        # Bandwidth savings should be proportional to pixel savings
        assert metrics.bandwidth_savings >= 0.0
        assert metrics.bandwidth_savings <= metrics.pixel_savings
