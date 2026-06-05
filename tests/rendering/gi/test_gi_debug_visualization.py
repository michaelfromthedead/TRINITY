"""Tests for GI debug visualization overlay (T-GIR-P10.1).

Tests cover:
- Debug mode control and @debug decorator
- Color utilities and gradient generation
- GIDebugConfig validation
- ProbeGridVisualization color mapping and rendering
- VoxelOccupancyVisualization wireframe and slice rendering
- SSGIConfidenceHeatmap generation and statistics
- PathTracerComparisonHeatmap accuracy and PSNR computation
- ReflectionTechniqueMask correctness
- GIDebugOverlay compositing and toggle behavior
- WGSL shader generation
- Zero overhead when debug disabled
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.gi.gi_debug_visualization import (
    # Debug mode control
    set_debug_mode,
    is_debug_enabled,
    debug,
    register_debug_pass,
    get_registered_debug_passes,
    clear_debug_passes,
    # Constants
    DEFAULT_OVERLAY_OPACITY,
    DEFAULT_DIFFERENCE_THRESHOLD,
    DEFAULT_CONFIDENCE_LOW,
    DEFAULT_PROBE_SPHERE_RADIUS,
    PSNR_EXCELLENT,
    PSNR_GOOD,
    PSNR_ACCEPTABLE,
    PSNR_POOR,
    # Color
    Color,
    COLOR_RED,
    COLOR_GREEN,
    COLOR_BLUE,
    COLOR_BLACK,
    COLOR_WHITE,
    TECHNIQUE_COLOR_RT,
    TECHNIQUE_COLOR_SSR,
    TECHNIQUE_COLOR_PROBES,
    TECHNIQUE_COLOR_ENV,
    create_heatmap_gradient,
    sample_gradient,
    HeatmapColorScale,
    # Enums
    ProbeColorMode,
    ProbeState,
    ReflectionTechnique,
    VoxelSliceAxis,
    # Config
    GIDebugConfig,
    # Data structures
    ProbeVisualizationData,
    VoxelData,
    ConfidencePixel,
    DifferencePixel,
    ComparisonStats,
    ReflectionPixel,
    # Visualization classes
    ProbeGridVisualization,
    VoxelOccupancyVisualization,
    SSGIConfidenceHeatmap,
    PathTracerComparisonHeatmap,
    ReflectionTechniqueMask,
    GIDebugOverlay,
    # WGSL generation
    generate_debug_overlay_wgsl,
    generate_probe_billboard_wgsl,
    generate_voxel_wireframe_wgsl,
    # Utilities
    estimate_debug_memory,
    create_test_probes,
    create_test_voxels,
)


# =============================================================================
# Debug Mode Tests
# =============================================================================


class TestDebugMode:
    """Tests for debug mode control."""

    def test_default_debug_enabled(self) -> None:
        """Test debug mode is enabled by default."""
        # Reset to default
        set_debug_mode(True)
        assert is_debug_enabled() is True

    def test_set_debug_mode_disabled(self) -> None:
        """Test disabling debug mode."""
        set_debug_mode(False)
        assert is_debug_enabled() is False
        # Reset for other tests
        set_debug_mode(True)

    def test_set_debug_mode_enabled(self) -> None:
        """Test enabling debug mode."""
        set_debug_mode(False)
        set_debug_mode(True)
        assert is_debug_enabled() is True


class TestDebugDecorator:
    """Tests for the @debug decorator."""

    def test_debug_decorator_when_enabled(self) -> None:
        """Test @debug decorated function runs when debug is enabled."""
        set_debug_mode(True)

        @debug
        def test_func() -> str:
            return "executed"

        assert test_func() == "executed"

    def test_debug_decorator_when_disabled(self) -> None:
        """Test @debug decorated function returns None when disabled."""
        set_debug_mode(False)

        @debug
        def test_func() -> str:
            return "executed"

        assert test_func() is None
        set_debug_mode(True)

    def test_debug_decorator_preserves_args(self) -> None:
        """Test @debug decorator passes arguments correctly."""
        set_debug_mode(True)

        @debug
        def add(a: int, b: int) -> int:
            return a + b

        assert add(3, 5) == 8

    def test_debug_decorator_preserves_kwargs(self) -> None:
        """Test @debug decorator passes kwargs correctly."""
        set_debug_mode(True)

        @debug
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        assert greet(name="World", greeting="Hi") == "Hi, World!"


class TestDebugPassRegistry:
    """Tests for debug pass registration."""

    def setup_method(self) -> None:
        """Clear registered passes before each test."""
        clear_debug_passes()
        set_debug_mode(True)

    def test_register_debug_pass(self) -> None:
        """Test registering a debug pass."""
        def my_pass() -> None:
            pass

        register_debug_pass("test_pass", my_pass)
        passes = get_registered_debug_passes()

        assert "test_pass" in passes
        assert passes["test_pass"] is my_pass

    def test_clear_debug_passes(self) -> None:
        """Test clearing all debug passes."""
        register_debug_pass("pass1", lambda: None)
        register_debug_pass("pass2", lambda: None)

        clear_debug_passes()

        assert len(get_registered_debug_passes()) == 0

    def test_register_pass_when_disabled(self) -> None:
        """Test that passes are not registered when debug is disabled."""
        set_debug_mode(False)
        register_debug_pass("no_register", lambda: None)

        passes = get_registered_debug_passes()
        assert "no_register" not in passes

        set_debug_mode(True)


# =============================================================================
# Color Tests
# =============================================================================


class TestColor:
    """Tests for the Color class."""

    def test_color_defaults(self) -> None:
        """Test default color values."""
        color = Color()
        assert color.r == 0.0
        assert color.g == 0.0
        assert color.b == 0.0
        assert color.a == 1.0

    def test_color_clamping(self) -> None:
        """Test color values are clamped to [0, 1]."""
        color = Color(1.5, -0.5, 0.5, 2.0)
        assert color.r == 1.0
        assert color.g == 0.0
        assert color.b == 0.5
        assert color.a == 1.0

    def test_color_to_tuple(self) -> None:
        """Test conversion to RGBA tuple."""
        color = Color(0.1, 0.2, 0.3, 0.4)
        assert color.to_tuple() == (0.1, 0.2, 0.3, 0.4)

    def test_color_to_rgb_tuple(self) -> None:
        """Test conversion to RGB tuple."""
        color = Color(0.1, 0.2, 0.3, 0.4)
        assert color.to_rgb_tuple() == (0.1, 0.2, 0.3)

    def test_color_luminance(self) -> None:
        """Test luminance calculation."""
        white = Color(1.0, 1.0, 1.0)
        assert white.luminance() == pytest.approx(1.0)

        black = Color(0.0, 0.0, 0.0)
        assert black.luminance() == pytest.approx(0.0)

        red = Color(1.0, 0.0, 0.0)
        assert red.luminance() == pytest.approx(0.2126)

    def test_color_blend(self) -> None:
        """Test color blending."""
        black = Color(0.0, 0.0, 0.0)
        white = Color(1.0, 1.0, 1.0)

        mid = black.blend(white, 0.5)
        assert mid.r == pytest.approx(0.5)
        assert mid.g == pytest.approx(0.5)
        assert mid.b == pytest.approx(0.5)

    def test_color_blend_clamping(self) -> None:
        """Test blend factor is clamped."""
        black = Color(0.0, 0.0, 0.0)
        white = Color(1.0, 1.0, 1.0)

        result = black.blend(white, 2.0)
        assert result.r == pytest.approx(1.0)

    def test_color_from_hex_6(self) -> None:
        """Test creating color from 6-digit hex."""
        color = Color.from_hex("#FF8000")
        assert color.r == pytest.approx(1.0)
        assert color.g == pytest.approx(0.502, abs=0.01)
        assert color.b == pytest.approx(0.0)
        assert color.a == pytest.approx(1.0)

    def test_color_from_hex_8(self) -> None:
        """Test creating color from 8-digit hex with alpha."""
        color = Color.from_hex("FF000080")
        assert color.r == pytest.approx(1.0)
        assert color.a == pytest.approx(0.502, abs=0.01)

    def test_color_from_hex_invalid(self) -> None:
        """Test invalid hex raises error."""
        with pytest.raises(ValueError):
            Color.from_hex("FF00")


class TestHeatmapGradient:
    """Tests for heatmap gradient generation."""

    def test_blue_red_gradient_length(self) -> None:
        """Test gradient has correct number of steps."""
        gradient = create_heatmap_gradient(HeatmapColorScale.BLUE_RED, 256)
        assert len(gradient) == 256

    def test_blue_red_gradient_endpoints(self) -> None:
        """Test blue-red gradient starts blue and ends red."""
        gradient = create_heatmap_gradient(HeatmapColorScale.BLUE_RED, 256)

        # Start should be blue-ish
        assert gradient[0].b > gradient[0].r

        # End should be red
        assert gradient[-1].r > gradient[-1].b
        assert gradient[-1].r > gradient[-1].g

    def test_green_red_gradient(self) -> None:
        """Test green-red gradient."""
        gradient = create_heatmap_gradient(HeatmapColorScale.GREEN_RED, 256)

        # Start should be green
        assert gradient[0].g > gradient[0].r

        # End should be red
        assert gradient[-1].r > gradient[-1].g

    def test_viridis_gradient(self) -> None:
        """Test viridis gradient generates valid colors."""
        gradient = create_heatmap_gradient(HeatmapColorScale.VIRIDIS, 256)

        for color in gradient:
            assert 0.0 <= color.r <= 1.0
            assert 0.0 <= color.g <= 1.0
            assert 0.0 <= color.b <= 1.0

    def test_sample_gradient_middle(self) -> None:
        """Test sampling gradient at middle value."""
        gradient = create_heatmap_gradient(HeatmapColorScale.BLUE_RED, 256)
        color = sample_gradient(gradient, 0.5)

        # Middle should be greenish-yellow
        assert color.g > 0.5

    def test_sample_gradient_clamping(self) -> None:
        """Test gradient sampling clamps out-of-range values."""
        gradient = create_heatmap_gradient(HeatmapColorScale.BLUE_RED, 256)

        low = sample_gradient(gradient, -1.0)
        high = sample_gradient(gradient, 2.0)

        assert low.to_tuple() == gradient[0].to_tuple()
        assert high.to_tuple() == gradient[-1].to_tuple()


# =============================================================================
# Configuration Tests
# =============================================================================


class TestGIDebugConfig:
    """Tests for GIDebugConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GIDebugConfig()

        assert config.show_probes is False
        assert config.show_voxels is False
        assert config.show_ssgi_confidence is False
        assert config.show_path_tracer_diff is False
        assert config.show_reflection_mask is False
        assert config.overlay_opacity == pytest.approx(DEFAULT_OVERLAY_OPACITY)
        assert config.difference_threshold == pytest.approx(DEFAULT_DIFFERENCE_THRESHOLD)

    def test_config_validation_opacity(self) -> None:
        """Test opacity validation."""
        with pytest.raises(ValueError, match="overlay_opacity"):
            GIDebugConfig(overlay_opacity=1.5)

        with pytest.raises(ValueError, match="overlay_opacity"):
            GIDebugConfig(overlay_opacity=-0.1)

    def test_config_validation_threshold(self) -> None:
        """Test difference threshold validation."""
        with pytest.raises(ValueError, match="difference_threshold"):
            GIDebugConfig(difference_threshold=2.0)

    def test_config_validation_radius(self) -> None:
        """Test probe radius validation."""
        with pytest.raises(ValueError, match="probe_sphere_radius"):
            GIDebugConfig(probe_sphere_radius=0.0)

        with pytest.raises(ValueError, match="probe_sphere_radius"):
            GIDebugConfig(probe_sphere_radius=-1.0)

    def test_config_validation_slice_depth(self) -> None:
        """Test voxel slice depth validation."""
        with pytest.raises(ValueError, match="voxel_slice_depth"):
            GIDebugConfig(voxel_slice_depth=1.5)

    def test_config_validation_confidence(self) -> None:
        """Test confidence threshold validation."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            GIDebugConfig(confidence_threshold=-0.1)

    def test_any_enabled_false(self) -> None:
        """Test any_enabled when all disabled."""
        config = GIDebugConfig()
        assert config.any_enabled is False

    def test_any_enabled_true(self) -> None:
        """Test any_enabled when one is enabled."""
        config = GIDebugConfig(show_probes=True)
        assert config.any_enabled is True

    def test_count_enabled(self) -> None:
        """Test counting enabled visualizations."""
        config = GIDebugConfig(
            show_probes=True,
            show_voxels=True,
            show_ssgi_confidence=True,
        )
        assert config.count_enabled() == 3


# =============================================================================
# ProbeVisualizationData Tests
# =============================================================================


class TestProbeVisualizationData:
    """Tests for ProbeVisualizationData."""

    def test_default_values(self) -> None:
        """Test default probe values."""
        probe = ProbeVisualizationData(position=(0.0, 0.0, 0.0))

        assert probe.irradiance == (0.0, 0.0, 0.0)
        assert probe.state == ProbeState.ACTIVE
        assert probe.depth == 0
        assert probe.blend_weight == pytest.approx(1.0)

    def test_luminance_calculation(self) -> None:
        """Test luminance from irradiance."""
        probe = ProbeVisualizationData(
            position=(0.0, 0.0, 0.0),
            irradiance=(1.0, 1.0, 1.0),
        )
        assert probe.get_luminance() == pytest.approx(1.0)

    def test_irradiance_magnitude(self) -> None:
        """Test irradiance magnitude calculation."""
        probe = ProbeVisualizationData(
            position=(0.0, 0.0, 0.0),
            irradiance=(1.0, 0.0, 0.0),
        )
        assert probe.get_irradiance_magnitude() == pytest.approx(1.0)

        probe2 = ProbeVisualizationData(
            position=(0.0, 0.0, 0.0),
            irradiance=(1.0, 1.0, 1.0),
        )
        assert probe2.get_irradiance_magnitude() == pytest.approx(math.sqrt(3))


# =============================================================================
# ProbeGridVisualization Tests
# =============================================================================


class TestProbeGridVisualization:
    """Tests for ProbeGridVisualization."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        set_debug_mode(True)
        self.config = GIDebugConfig(show_probes=True)
        self.vis = ProbeGridVisualization(self.config)

    def test_initial_state(self) -> None:
        """Test initial visualization state."""
        assert self.vis.probe_count == 0

    def test_set_probes(self) -> None:
        """Test setting probe data."""
        probes = create_test_probes(grid_size=(2, 2, 1))
        self.vis.set_probes(probes)

        assert self.vis.probe_count == 4

    def test_probe_color_irradiance_mode(self) -> None:
        """Test probe color in irradiance mode."""
        self.vis.set_color_mode(ProbeColorMode.IRRADIANCE)

        low_probe = ProbeVisualizationData(
            position=(0.0, 0.0, 0.0),
            irradiance=(0.0, 0.0, 0.0),
        )
        high_probe = ProbeVisualizationData(
            position=(1.0, 0.0, 0.0),
            irradiance=(1.0, 1.0, 1.0),
        )

        self.vis.set_probes([low_probe, high_probe])

        low_color = self.vis.get_probe_color(low_probe)
        high_color = self.vis.get_probe_color(high_probe)

        # Low irradiance should be blue-ish (cold)
        # High irradiance should be red-ish (hot)
        assert low_color.b >= low_color.r or low_color.luminance() < 0.5
        assert high_color.r >= high_color.b

    def test_probe_color_state_mode(self) -> None:
        """Test probe color in state mode."""
        self.vis.set_color_mode(ProbeColorMode.STATE)

        active_probe = ProbeVisualizationData(
            position=(0.0, 0.0, 0.0),
            state=ProbeState.ACTIVE,
        )
        invalid_probe = ProbeVisualizationData(
            position=(1.0, 0.0, 0.0),
            state=ProbeState.INVALID,
        )

        active_color = self.vis.get_probe_color(active_probe)
        invalid_color = self.vis.get_probe_color(invalid_probe)

        # Active should be green
        assert active_color.g > active_color.r

        # Invalid should be red
        assert invalid_color.r > invalid_color.g

    def test_render_probes(self) -> None:
        """Test rendering probes returns correct data."""
        probes = create_test_probes(grid_size=(2, 2, 1))
        self.vis.set_probes(probes)

        result = self.vis.render_probes()

        assert result is not None
        assert len(result) == 4

        for pos, color, radius in result:
            assert len(pos) == 3
            assert isinstance(color, Color)
            assert radius > 0

    def test_render_probes_disabled(self) -> None:
        """Test render returns None when debug disabled."""
        set_debug_mode(False)

        probes = create_test_probes(grid_size=(2, 2, 1))
        self.vis.set_probes(probes)

        result = self.vis.render_probes()
        assert result is None

        set_debug_mode(True)

    def test_get_probe_at_index(self) -> None:
        """Test getting probe at specific index."""
        probes = create_test_probes(grid_size=(2, 2, 1))
        self.vis.set_probes(probes)

        probe = self.vis.get_probe_at_index(0)
        assert probe is not None

        invalid = self.vis.get_probe_at_index(100)
        assert invalid is None


# =============================================================================
# VoxelOccupancyVisualization Tests
# =============================================================================


class TestVoxelOccupancyVisualization:
    """Tests for VoxelOccupancyVisualization."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        set_debug_mode(True)
        self.config = GIDebugConfig(show_voxels=True)
        self.vis = VoxelOccupancyVisualization(self.config)

    def test_initial_state(self) -> None:
        """Test initial visualization state."""
        assert self.vis.voxel_count == 0
        assert self.vis.total_voxels == 64 * 64 * 64

    def test_set_voxels(self) -> None:
        """Test setting voxel data."""
        voxels = create_test_voxels(resolution=(8, 8, 8), fill_ratio=0.5)
        self.vis.set_voxels(voxels, (8, 8, 8))

        assert self.vis.voxel_count > 0

    def test_occupancy_ratio(self) -> None:
        """Test occupancy ratio calculation."""
        voxels = [VoxelData(position=(0, 0, 0), density=1.0)]
        self.vis.set_voxels(voxels, (2, 2, 2))

        assert self.vis.occupancy_ratio == pytest.approx(1 / 8)

    def test_set_slice_depth(self) -> None:
        """Test setting slice depth."""
        self.vis.set_slice_depth(0.75)
        assert self.config.voxel_slice_depth == pytest.approx(0.75)

    def test_set_slice_depth_clamping(self) -> None:
        """Test slice depth clamping."""
        self.vis.set_slice_depth(2.0)
        assert self.config.voxel_slice_depth == pytest.approx(1.0)

        self.vis.set_slice_depth(-1.0)
        assert self.config.voxel_slice_depth == pytest.approx(0.0)

    def test_render_wireframe(self) -> None:
        """Test wireframe rendering."""
        voxels = [VoxelData(position=(0, 0, 0), density=1.0)]
        self.vis.set_voxels(voxels, (4, 4, 4))

        result = self.vis.render_wireframe()

        assert result is not None
        # 12 edges per cube
        assert len(result) == 12

    def test_render_wireframe_empty(self) -> None:
        """Test wireframe with no voxels."""
        self.vis.set_voxels([], (4, 4, 4))
        result = self.vis.render_wireframe()

        assert result is not None
        assert len(result) == 0

    def test_render_slice(self) -> None:
        """Test slice rendering."""
        # Create voxels at different Y levels
        voxels = [
            VoxelData(position=(0, 0, 0), density=1.0),
            VoxelData(position=(1, 0, 0), density=0.5),
            VoxelData(position=(0, 1, 0), density=1.0),  # Different Y
        ]
        self.vis.set_voxels(voxels, (4, 4, 4))

        self.config.voxel_slice_axis = VoxelSliceAxis.Y
        self.vis.set_slice_depth(0.0)  # Y=0 slice

        result = self.vis.render_slice()

        assert result is not None
        # Should have 2 voxels at Y=0
        assert len(result) == 2

    def test_render_wireframe_disabled(self) -> None:
        """Test wireframe returns None when debug disabled."""
        set_debug_mode(False)

        voxels = [VoxelData(position=(0, 0, 0), density=1.0)]
        self.vis.set_voxels(voxels, (4, 4, 4))

        result = self.vis.render_wireframe()
        assert result is None

        set_debug_mode(True)


# =============================================================================
# SSGIConfidenceHeatmap Tests
# =============================================================================


class TestSSGIConfidenceHeatmap:
    """Tests for SSGIConfidenceHeatmap."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        set_debug_mode(True)
        self.config = GIDebugConfig(show_ssgi_confidence=True)
        self.vis = SSGIConfidenceHeatmap(self.config)

    def test_set_confidence_data(self) -> None:
        """Test setting confidence data."""
        pixels = [
            ConfidencePixel(x=0, y=0, confidence=0.9),
            ConfidencePixel(x=1, y=0, confidence=0.5),
        ]
        self.vis.set_confidence_data(pixels, 2, 1)

        stats = self.vis.get_statistics()
        assert stats["max"] == pytest.approx(0.9)
        assert stats["min"] == pytest.approx(0.5)

    def test_confidence_color_high(self) -> None:
        """Test high confidence color is greenish."""
        color = self.vis.get_confidence_color(0.9)

        # High confidence (inverted) should be green
        assert color.g > color.r

    def test_confidence_color_low(self) -> None:
        """Test low confidence color is reddish."""
        color = self.vis.get_confidence_color(0.1)

        # Low confidence (inverted) should be red
        assert color.r > color.g

    def test_set_threshold(self) -> None:
        """Test setting confidence threshold."""
        self.vis.set_threshold(0.7)
        assert self.config.confidence_threshold == pytest.approx(0.7)

    def test_render_heatmap(self) -> None:
        """Test heatmap rendering."""
        pixels = [
            ConfidencePixel(x=0, y=0, confidence=0.9),
            ConfidencePixel(x=1, y=0, confidence=0.1),
        ]
        self.vis.set_confidence_data(pixels, 2, 1)

        result = self.vis.render_heatmap()

        assert result is not None
        assert len(result) == 2

    def test_render_threshold_mask(self) -> None:
        """Test threshold mask rendering."""
        pixels = [
            ConfidencePixel(x=0, y=0, confidence=0.9),
            ConfidencePixel(x=1, y=0, confidence=0.1),
        ]
        self.vis.set_confidence_data(pixels, 2, 1)
        self.vis.set_threshold(0.5)

        result = self.vis.render_threshold_mask()

        assert result is not None
        # Only pixel with confidence < 0.5 should be in mask
        assert len(result) == 1
        assert result[0] == (1, 0)

    def test_statistics(self) -> None:
        """Test confidence statistics calculation."""
        pixels = [
            ConfidencePixel(x=0, y=0, confidence=0.2),
            ConfidencePixel(x=1, y=0, confidence=0.4),
            ConfidencePixel(x=2, y=0, confidence=0.6),
            ConfidencePixel(x=3, y=0, confidence=0.8),
        ]
        self.vis.set_confidence_data(pixels, 4, 1)
        self.vis.set_threshold(0.5)

        stats = self.vis.get_statistics()

        assert stats["min"] == pytest.approx(0.2)
        assert stats["max"] == pytest.approx(0.8)
        assert stats["mean"] == pytest.approx(0.5)
        # 2 out of 4 below threshold (0.2 and 0.4)
        assert stats["below_threshold_ratio"] == pytest.approx(0.5)

    def test_empty_statistics(self) -> None:
        """Test statistics with no data."""
        stats = self.vis.get_statistics()

        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0


# =============================================================================
# PathTracerComparisonHeatmap Tests
# =============================================================================


class TestPathTracerComparisonHeatmap:
    """Tests for PathTracerComparisonHeatmap."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        set_debug_mode(True)
        self.config = GIDebugConfig(show_path_tracer_diff=True)
        self.vis = PathTracerComparisonHeatmap(self.config)

    def test_difference_pixel_auto_compute(self) -> None:
        """Test DifferencePixel auto-computes difference."""
        pixel = DifferencePixel(
            x=0, y=0,
            gi_color=(0.5, 0.5, 0.5),
            reference_color=(0.6, 0.6, 0.6),
        )

        assert pixel.difference > 0

    def test_set_comparison_data(self) -> None:
        """Test setting comparison data."""
        pixels = [
            DifferencePixel(
                x=0, y=0,
                gi_color=(0.5, 0.5, 0.5),
                reference_color=(0.5, 0.5, 0.5),
            ),
        ]
        self.vis.set_comparison_data(pixels, 1, 1)

        stats = self.vis.compute_stats()
        assert stats.rmse == pytest.approx(0.0)

    def test_compute_stats_rmse(self) -> None:
        """Test RMSE computation."""
        pixels = [
            DifferencePixel(
                x=0, y=0,
                gi_color=(0.0, 0.0, 0.0),
                reference_color=(0.1, 0.1, 0.1),
            ),
        ]
        self.vis.set_comparison_data(pixels, 1, 1)

        stats = self.vis.compute_stats()
        assert stats.rmse == pytest.approx(0.1)

    def test_compute_stats_psnr(self) -> None:
        """Test PSNR computation."""
        # Perfect match should have infinite PSNR
        pixels = [
            DifferencePixel(
                x=0, y=0,
                gi_color=(0.5, 0.5, 0.5),
                reference_color=(0.5, 0.5, 0.5),
            ),
        ]
        self.vis.set_comparison_data(pixels, 1, 1)

        stats = self.vis.compute_stats()
        assert stats.psnr == float("inf")

    def test_difference_color_low(self) -> None:
        """Test low difference color is blue/dark."""
        color = self.vis.get_difference_color(0.0)

        # Zero difference should be at gradient start (blue)
        assert color.b > color.r or color.luminance() < 0.5

    def test_difference_color_high(self) -> None:
        """Test high difference color is red."""
        color = self.vis.get_difference_color(0.5)

        # High difference should be red
        assert color.r > color.b

    def test_render_difference(self) -> None:
        """Test difference heatmap rendering."""
        pixels = [
            DifferencePixel(
                x=0, y=0,
                gi_color=(0.5, 0.5, 0.5),
                reference_color=(0.6, 0.6, 0.6),
            ),
        ]
        self.vis.set_comparison_data(pixels, 1, 1)

        result = self.vis.render_difference()

        assert result is not None
        assert len(result) == 1

    def test_psnr_quality_excellent(self) -> None:
        """Test PSNR quality assessment - excellent."""
        pixels = [
            DifferencePixel(
                x=0, y=0,
                gi_color=(0.5, 0.5, 0.5),
                reference_color=(0.5, 0.5, 0.5),
            ),
        ]
        self.vis.set_comparison_data(pixels, 1, 1)

        quality = self.vis.get_psnr_quality()
        assert quality == "excellent"

    def test_above_threshold_ratio(self) -> None:
        """Test above threshold ratio in stats."""
        self.config.difference_threshold = 0.1

        pixels = [
            DifferencePixel(
                x=0, y=0,
                gi_color=(0.5, 0.5, 0.5),
                reference_color=(0.5, 0.5, 0.5),
                difference=0.05,  # Below threshold
            ),
            DifferencePixel(
                x=1, y=0,
                gi_color=(0.5, 0.5, 0.5),
                reference_color=(0.7, 0.7, 0.7),
                difference=0.2,  # Above threshold
            ),
        ]
        self.vis.set_comparison_data(pixels, 2, 1)

        stats = self.vis.compute_stats()
        assert stats.above_threshold_ratio == pytest.approx(0.5)


# =============================================================================
# ReflectionTechniqueMask Tests
# =============================================================================


class TestReflectionTechniqueMask:
    """Tests for ReflectionTechniqueMask."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        set_debug_mode(True)
        self.config = GIDebugConfig(show_reflection_mask=True)
        self.vis = ReflectionTechniqueMask(self.config)

    def test_technique_colors(self) -> None:
        """Test technique color mapping."""
        rt_color = self.vis.get_technique_color(ReflectionTechnique.RAY_TRACED)
        assert rt_color == TECHNIQUE_COLOR_RT

        ssr_color = self.vis.get_technique_color(ReflectionTechnique.SSR)
        assert ssr_color == TECHNIQUE_COLOR_SSR

        probe_color = self.vis.get_technique_color(ReflectionTechnique.PROBES)
        assert probe_color == TECHNIQUE_COLOR_PROBES

        env_color = self.vis.get_technique_color(ReflectionTechnique.ENVIRONMENT)
        assert env_color == TECHNIQUE_COLOR_ENV

    def test_render_mask(self) -> None:
        """Test mask rendering."""
        pixels = [
            ReflectionPixel(x=0, y=0, technique=ReflectionTechnique.RAY_TRACED),
            ReflectionPixel(x=1, y=0, technique=ReflectionTechnique.SSR),
        ]
        self.vis.set_technique_data(pixels, 2, 1)

        result = self.vis.render_mask()

        assert result is not None
        assert len(result) == 2

    def test_technique_coverage(self) -> None:
        """Test technique coverage calculation."""
        pixels = [
            ReflectionPixel(x=0, y=0, technique=ReflectionTechnique.RAY_TRACED),
            ReflectionPixel(x=1, y=0, technique=ReflectionTechnique.RAY_TRACED),
            ReflectionPixel(x=2, y=0, technique=ReflectionTechnique.SSR),
            ReflectionPixel(x=3, y=0, technique=ReflectionTechnique.PROBES),
        ]
        self.vis.set_technique_data(pixels, 4, 1)

        coverage = self.vis.get_technique_coverage()

        assert coverage[ReflectionTechnique.RAY_TRACED] == pytest.approx(0.5)
        assert coverage[ReflectionTechnique.SSR] == pytest.approx(0.25)
        assert coverage[ReflectionTechnique.PROBES] == pytest.approx(0.25)
        assert coverage[ReflectionTechnique.ENVIRONMENT] == pytest.approx(0.0)

    def test_transition_pixels(self) -> None:
        """Test finding technique transition pixels."""
        # 2x2 grid with different techniques
        pixels = [
            ReflectionPixel(x=0, y=0, technique=ReflectionTechnique.RAY_TRACED),
            ReflectionPixel(x=1, y=0, technique=ReflectionTechnique.SSR),
            ReflectionPixel(x=0, y=1, technique=ReflectionTechnique.RAY_TRACED),
            ReflectionPixel(x=1, y=1, technique=ReflectionTechnique.SSR),
        ]
        self.vis.set_technique_data(pixels, 2, 2)

        transitions = self.vis.get_transition_pixels()

        # Pixels at x=0 should be transitions (adjacent to SSR)
        # Pixels at x=1 should be transitions (adjacent to RT)
        assert len(transitions) == 4  # All are at boundaries


# =============================================================================
# GIDebugOverlay Tests
# =============================================================================


class TestGIDebugOverlay:
    """Tests for GIDebugOverlay."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        set_debug_mode(True)
        self.overlay = GIDebugOverlay()

    def test_default_config(self) -> None:
        """Test overlay initializes with default config."""
        assert self.overlay.config.show_probes is False
        assert self.overlay.config.show_voxels is False

    def test_toggle_visualization(self) -> None:
        """Test toggling a visualization."""
        result = self.overlay.toggle("probes")
        assert result is True
        assert self.overlay.config.show_probes is True

        result = self.overlay.toggle("probes")
        assert result is False
        assert self.overlay.config.show_probes is False

    def test_toggle_explicit(self) -> None:
        """Test explicit toggle state."""
        self.overlay.toggle("probes", enabled=True)
        assert self.overlay.config.show_probes is True

        self.overlay.toggle("probes", enabled=True)  # Already True
        assert self.overlay.config.show_probes is True

    def test_toggle_invalid(self) -> None:
        """Test toggling invalid visualization raises error."""
        with pytest.raises(ValueError, match="Unknown visualization"):
            self.overlay.toggle("invalid_vis")

    def test_set_opacity(self) -> None:
        """Test setting overlay opacity."""
        self.overlay.set_opacity(0.75)
        assert self.overlay.config.overlay_opacity == pytest.approx(0.75)

    def test_set_opacity_clamping(self) -> None:
        """Test opacity clamping."""
        self.overlay.set_opacity(2.0)
        assert self.overlay.config.overlay_opacity == pytest.approx(1.0)

        self.overlay.set_opacity(-1.0)
        assert self.overlay.config.overlay_opacity == pytest.approx(0.0)

    def test_enable_all(self) -> None:
        """Test enabling all visualizations."""
        self.overlay.enable_all()

        assert self.overlay.config.show_probes is True
        assert self.overlay.config.show_voxels is True
        assert self.overlay.config.show_ssgi_confidence is True
        assert self.overlay.config.show_path_tracer_diff is True
        assert self.overlay.config.show_reflection_mask is True

    def test_disable_all(self) -> None:
        """Test disabling all visualizations."""
        self.overlay.enable_all()
        self.overlay.disable_all()

        assert self.overlay.config.show_probes is False
        assert self.overlay.config.show_voxels is False
        assert self.overlay.config.show_ssgi_confidence is False
        assert self.overlay.config.show_path_tracer_diff is False
        assert self.overlay.config.show_reflection_mask is False

    def test_render_all_disabled(self) -> None:
        """Test render_all with all disabled."""
        result = self.overlay.render_all()

        assert result is not None
        assert result["probes"] is None
        assert result["voxels_wireframe"] is None

    def test_render_all_enabled(self) -> None:
        """Test render_all with probes enabled."""
        self.overlay.toggle("probes", enabled=True)
        probes = create_test_probes(grid_size=(2, 2, 1))
        self.overlay.probe_visualization.set_probes(probes)

        result = self.overlay.render_all()

        assert result is not None
        assert result["probes"] is not None
        assert len(result["probes"]) == 4

    def test_get_all_statistics(self) -> None:
        """Test getting all statistics."""
        probes = create_test_probes(grid_size=(2, 2, 1))
        self.overlay.probe_visualization.set_probes(probes)

        stats = self.overlay.get_all_statistics()

        assert "probes" in stats
        assert stats["probes"]["count"] == 4
        assert "voxels" in stats
        assert "ssgi" in stats
        assert "comparison" in stats
        assert "reflection" in stats

    def test_render_all_debug_disabled(self) -> None:
        """Test render_all returns None when debug disabled."""
        set_debug_mode(False)

        result = self.overlay.render_all()
        assert result is None

        set_debug_mode(True)


# =============================================================================
# WGSL Generation Tests
# =============================================================================


class TestWGSLGeneration:
    """Tests for WGSL shader generation."""

    def test_debug_overlay_wgsl(self) -> None:
        """Test debug overlay WGSL generation."""
        wgsl = generate_debug_overlay_wgsl()

        assert "DebugConfig" in wgsl
        assert "overlay_opacity" in wgsl
        assert "@compute" in wgsl
        assert "@workgroup_size" in wgsl

    def test_probe_billboard_wgsl(self) -> None:
        """Test probe billboard WGSL generation."""
        wgsl = generate_probe_billboard_wgsl()

        assert "ProbeData" in wgsl
        assert "CameraData" in wgsl
        assert "@vertex" in wgsl
        assert "@fragment" in wgsl

    def test_voxel_wireframe_wgsl(self) -> None:
        """Test voxel wireframe WGSL generation."""
        wgsl = generate_voxel_wireframe_wgsl()

        assert "LineVertex" in wgsl
        assert "CameraData" in wgsl
        assert "@vertex" in wgsl
        assert "@fragment" in wgsl


# =============================================================================
# Utility Tests
# =============================================================================


class TestUtilities:
    """Tests for utility functions."""

    def test_estimate_debug_memory(self) -> None:
        """Test debug memory estimation."""
        memory = estimate_debug_memory(
            probe_count=1000,
            voxel_count=5000,
            width=1920,
            height=1080,
        )

        assert memory > 0
        # Should be significant for these parameters
        assert memory > 1_000_000  # > 1MB

    def test_create_test_probes(self) -> None:
        """Test creating test probes."""
        probes = create_test_probes(grid_size=(3, 3, 2))

        assert len(probes) == 18  # 3 * 3 * 2

        # Check positions are spaced correctly
        assert probes[0].position == (0.0, 0.0, 0.0)
        assert probes[1].position == (2.0, 0.0, 0.0)  # Default spacing = 2.0

    def test_create_test_voxels(self) -> None:
        """Test creating test voxels."""
        voxels = create_test_voxels(resolution=(8, 8, 8), fill_ratio=0.5)

        expected_count = int(8 * 8 * 8 * 0.5)
        assert len(voxels) == expected_count

        # All voxels should have valid positions
        for voxel in voxels:
            assert 0 <= voxel.position[0] < 8
            assert 0 <= voxel.position[1] < 8
            assert 0 <= voxel.position[2] < 8


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the debug visualization system."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        set_debug_mode(True)

    def test_full_overlay_workflow(self) -> None:
        """Test complete overlay workflow."""
        # Create overlay
        config = GIDebugConfig(
            show_probes=True,
            show_ssgi_confidence=True,
            overlay_opacity=0.7,
        )
        overlay = GIDebugOverlay(config)

        # Set up probe data
        probes = create_test_probes(grid_size=(4, 4, 2))
        overlay.probe_visualization.set_probes(probes)

        # Set up SSGI confidence data
        ssgi_pixels = [
            ConfidencePixel(x=i % 10, y=i // 10, confidence=i / 100.0)
            for i in range(100)
        ]
        overlay.ssgi_visualization.set_confidence_data(ssgi_pixels, 10, 10)

        # Render all
        result = overlay.render_all()

        assert result is not None
        assert result["probes"] is not None
        assert len(result["probes"]) == 32
        assert result["ssgi_heatmap"] is not None
        assert len(result["ssgi_heatmap"]) == 100

        # Check statistics
        stats = overlay.get_all_statistics()
        assert stats["probes"]["count"] == 32
        assert stats["ssgi"]["mean"] == pytest.approx(0.495)

    def test_zero_overhead_when_disabled(self) -> None:
        """Test that debug code has zero overhead when disabled."""
        set_debug_mode(False)

        overlay = GIDebugOverlay()
        overlay.enable_all()

        # Set up lots of data
        probes = create_test_probes(grid_size=(10, 10, 5))
        overlay.probe_visualization.set_probes(probes)

        # Render should return None immediately
        result = overlay.render_all()
        assert result is None

        # Individual renders should also return None
        assert overlay.probe_visualization.render_probes() is None
        assert overlay.voxel_visualization.render_wireframe() is None
        assert overlay.ssgi_visualization.render_heatmap() is None

        set_debug_mode(True)

    def test_comparison_with_path_tracer(self) -> None:
        """Test path tracer comparison workflow."""
        overlay = GIDebugOverlay()
        overlay.toggle("comparison", enabled=True)

        # Simulate GI vs path tracer comparison
        # Most pixels match well, some have errors
        pixels = []
        for y in range(10):
            for x in range(10):
                if x < 8 and y < 8:
                    # Good match
                    gi = (0.5, 0.5, 0.5)
                    ref = (0.5, 0.5, 0.5)
                else:
                    # Error region (>10% difference)
                    gi = (0.3, 0.3, 0.3)
                    ref = (0.5, 0.5, 0.5)

                pixels.append(DifferencePixel(x=x, y=y, gi_color=gi, reference_color=ref))

        overlay.comparison_visualization.set_comparison_data(pixels, 10, 10)

        stats = overlay.comparison_visualization.compute_stats()

        # 64/100 pixels match, 36/100 have errors
        assert stats.above_threshold_ratio == pytest.approx(0.36)

        # Quality should not be excellent due to errors
        quality = overlay.comparison_visualization.get_psnr_quality()
        assert quality in ["good", "acceptable", "poor"]
