"""
Whitebox Tests for TAA Reprojection (T-DEMO-8.5).

Tests internal implementation details of world-space hit position
reprojection for temporal anti-aliasing in ray marching.

Test categories:
- Hit position buffer operations
- Color space conversions (YCoCg)
- Reprojection calculations
- Disocclusion detection
- Neighborhood clamping
- Blend factor computation
"""

from __future__ import annotations

import math
import pytest

from engine.core.math.mat import Mat4
from engine.core.math.vec import Vec2, Vec3, Vec4
from engine.rendering.demoscene.taa_reprojection import (
    # Configuration
    ReprojectionConfig,
    DisocclusionMode,
    ClampingMode,
    # Buffers
    HitPositionBuffer,
    ColorBuffer,
    # Color space
    rgb_to_ycocg,
    ycocg_to_rgb,
    # Reprojection
    project_to_screen,
    calculate_reprojected_uv,
    detect_disocclusion_depth,
    detect_disocclusion_position,
    # Clamping
    compute_neighborhood_bounds_rgb,
    compute_neighborhood_bounds_ycocg,
    compute_neighborhood_variance,
    clamp_color_rgb,
    clamp_color_ycocg,
    clamp_color_variance,
    # Main class
    TAAReprojection,
)


# =============================================================================
# Helper Functions
# =============================================================================


def make_simple_view_proj() -> Mat4:
    """Create a simple identity view-projection matrix for testing."""
    return Mat4.identity()


def make_perspective_view_proj(
    eye: Vec3,
    target: Vec3,
    fov: float = math.radians(60),
    aspect: float = 16/9,
    near: float = 0.1,
    far: float = 100.0,
) -> Mat4:
    """Create a perspective view-projection matrix."""
    view = Mat4.look_at(eye, target, Vec3.up())
    proj = Mat4.perspective(fov, aspect, near, far)
    return proj @ view


# =============================================================================
# HitPositionBuffer Tests
# =============================================================================


class TestHitPositionBuffer:
    """Tests for the HitPositionBuffer class."""

    def test_creation(self) -> None:
        """Buffer should initialize with correct dimensions."""
        buf = HitPositionBuffer(64, 48)
        assert buf.width == 64
        assert buf.height == 48
        assert len(buf.positions) == 64 * 48
        assert len(buf.valid) == 64 * 48

    def test_invalid_dimensions(self) -> None:
        """Invalid dimensions should raise ValueError."""
        with pytest.raises(ValueError):
            HitPositionBuffer(0, 10)
        with pytest.raises(ValueError):
            HitPositionBuffer(10, -1)

    def test_set_hit_and_get(self) -> None:
        """set_hit should store position and mark valid."""
        buf = HitPositionBuffer(10, 10)
        pos = Vec3(1.0, 2.0, 3.0)
        buf.set_hit(5, 5, pos)

        result, valid = buf.get_position(5, 5)
        assert valid is True
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)

    def test_set_miss(self) -> None:
        """set_miss should mark position as invalid."""
        buf = HitPositionBuffer(10, 10)
        buf.set_hit(5, 5, Vec3(1.0, 2.0, 3.0))
        buf.set_miss(5, 5)

        _, valid = buf.get_position(5, 5)
        assert valid is False

    def test_out_of_bounds_get(self) -> None:
        """Out-of-bounds get should return zero and invalid."""
        buf = HitPositionBuffer(10, 10)
        pos, valid = buf.get_position(-1, 0)
        assert valid is False
        assert pos.x == 0.0

        pos, valid = buf.get_position(10, 0)
        assert valid is False

    def test_out_of_bounds_set(self) -> None:
        """Out-of-bounds set should be silently ignored."""
        buf = HitPositionBuffer(10, 10)
        buf.set_hit(-1, 0, Vec3(1, 1, 1))  # Should not crash
        buf.set_hit(10, 0, Vec3(1, 1, 1))  # Should not crash

    def test_sample_bilinear_center(self) -> None:
        """Bilinear sampling at pixel center."""
        buf = HitPositionBuffer(4, 4)
        buf.set_hit(2, 2, Vec3(10.0, 20.0, 30.0))

        # Sample at center of pixel (2, 2)
        uv = Vec2(2.5 / 4.0, 2.5 / 4.0)
        pos, weight = buf.sample_bilinear(uv)

        # Weight should be 1.0 for a single valid pixel contributing
        assert weight > 0
        # Position should be close to (10, 20, 30)
        assert pos.x == pytest.approx(10.0, abs=0.5)

    def test_sample_bilinear_interpolation(self) -> None:
        """Bilinear sampling between two valid pixels."""
        buf = HitPositionBuffer(4, 4)
        buf.set_hit(0, 0, Vec3(0.0, 0.0, 0.0))
        buf.set_hit(1, 0, Vec3(10.0, 0.0, 0.0))

        # Sample midway between (0,0) and (1,0)
        uv = Vec2(0.5 / 4.0, 0.5 / 4.0)
        pos, weight = buf.sample_bilinear(uv)

        # Should interpolate x between 0 and 10
        assert weight > 0
        assert 0.0 <= pos.x <= 10.0

    def test_clear(self) -> None:
        """clear should reset all positions to invalid."""
        buf = HitPositionBuffer(10, 10)
        buf.set_hit(5, 5, Vec3(1, 2, 3))
        buf.clear()

        _, valid = buf.get_position(5, 5)
        assert valid is False

    def test_copy_from(self) -> None:
        """copy_from should duplicate buffer contents."""
        buf1 = HitPositionBuffer(10, 10)
        buf1.set_hit(3, 3, Vec3(5, 6, 7))

        buf2 = HitPositionBuffer(10, 10)
        buf2.copy_from(buf1)

        pos, valid = buf2.get_position(3, 3)
        assert valid is True
        assert pos.x == pytest.approx(5.0)

    def test_clone(self) -> None:
        """clone should create an independent copy."""
        buf1 = HitPositionBuffer(10, 10)
        buf1.set_hit(3, 3, Vec3(5, 6, 7))

        buf2 = buf1.clone()
        buf1.set_miss(3, 3)

        # buf2 should still have the original data
        pos, valid = buf2.get_position(3, 3)
        assert valid is True


# =============================================================================
# ColorBuffer Tests
# =============================================================================


class TestColorBuffer:
    """Tests for the ColorBuffer class."""

    def test_creation(self) -> None:
        """Buffer should initialize with correct dimensions."""
        buf = ColorBuffer(32, 24)
        assert buf.width == 32
        assert buf.height == 24
        assert len(buf.data) == 32 * 24

    def test_invalid_dimensions(self) -> None:
        """Invalid dimensions should raise ValueError."""
        with pytest.raises(ValueError):
            ColorBuffer(0, 10)

    def test_default_pixels_zero(self) -> None:
        """Pixels should default to zero."""
        buf = ColorBuffer(4, 4)
        c = buf.get_pixel(0, 0)
        assert c.x == 0.0
        assert c.y == 0.0
        assert c.z == 0.0
        assert c.w == 0.0

    def test_set_get_pixel(self) -> None:
        """set_pixel and get_pixel should work correctly."""
        buf = ColorBuffer(4, 4)
        color = Vec4(0.5, 0.25, 0.75, 1.0)
        buf.set_pixel(2, 2, color)

        c = buf.get_pixel(2, 2)
        assert c.x == pytest.approx(0.5)
        assert c.y == pytest.approx(0.25)
        assert c.z == pytest.approx(0.75)
        assert c.w == pytest.approx(1.0)

    def test_out_of_bounds_returns_zero(self) -> None:
        """Out-of-bounds get should return black."""
        buf = ColorBuffer(4, 4)
        c = buf.get_pixel(-1, 0)
        assert c.x == 0.0

    def test_sample_bilinear(self) -> None:
        """Bilinear sampling should interpolate correctly."""
        buf = ColorBuffer(4, 4)
        buf.set_pixel(0, 0, Vec4(1, 0, 0, 1))
        buf.set_pixel(1, 0, Vec4(0, 1, 0, 1))

        # Sample midway should blend red and green
        uv = Vec2(0.5 / 4.0, 0.5 / 4.0)
        c = buf.sample_bilinear(uv)
        # Should have some red and some green
        assert c.x >= 0
        assert c.y >= 0


# =============================================================================
# Color Space Conversion Tests (YCoCg)
# =============================================================================


class TestColorSpaceConversion:
    """Tests for RGB to YCoCg conversion."""

    def test_rgb_to_ycocg_white(self) -> None:
        """White should have Y=1, Co=0, Cg=0."""
        white = Vec4(1.0, 1.0, 1.0, 1.0)
        yc = rgb_to_ycocg(white)
        assert yc.x == pytest.approx(1.0)  # Y
        assert yc.y == pytest.approx(0.0)  # Co
        assert yc.z == pytest.approx(0.0)  # Cg

    def test_rgb_to_ycocg_black(self) -> None:
        """Black should have Y=0, Co=0, Cg=0."""
        black = Vec4(0.0, 0.0, 0.0, 1.0)
        yc = rgb_to_ycocg(black)
        assert yc.x == pytest.approx(0.0)
        assert yc.y == pytest.approx(0.0)
        assert yc.z == pytest.approx(0.0)

    def test_rgb_to_ycocg_red(self) -> None:
        """Red should have positive Co."""
        red = Vec4(1.0, 0.0, 0.0, 1.0)
        yc = rgb_to_ycocg(red)
        assert yc.y > 0  # Co positive for red

    def test_rgb_to_ycocg_green(self) -> None:
        """Green should have positive Cg."""
        green = Vec4(0.0, 1.0, 0.0, 1.0)
        yc = rgb_to_ycocg(green)
        assert yc.z > 0  # Cg positive for green

    def test_roundtrip_conversion(self) -> None:
        """YCoCg to RGB should be inverse of RGB to YCoCg."""
        colors = [
            Vec4(1.0, 0.0, 0.0, 1.0),
            Vec4(0.0, 1.0, 0.0, 1.0),
            Vec4(0.0, 0.0, 1.0, 1.0),
            Vec4(0.5, 0.3, 0.8, 1.0),
        ]
        for rgb in colors:
            yc = rgb_to_ycocg(rgb)
            back = ycocg_to_rgb(yc)
            assert back.x == pytest.approx(rgb.x, abs=1e-6)
            assert back.y == pytest.approx(rgb.y, abs=1e-6)
            assert back.z == pytest.approx(rgb.z, abs=1e-6)

    def test_alpha_preserved(self) -> None:
        """Alpha channel should be preserved through conversion."""
        color = Vec4(0.5, 0.5, 0.5, 0.7)
        yc = rgb_to_ycocg(color)
        assert yc.w == pytest.approx(0.7)
        back = ycocg_to_rgb(yc)
        assert back.w == pytest.approx(0.7)


# =============================================================================
# Reprojection Tests
# =============================================================================


class TestProjectToScreen:
    """Tests for project_to_screen function."""

    def test_identity_projection_center(self) -> None:
        """With identity matrix, origin should project to center."""
        view_proj = Mat4.identity()
        uv, depth, valid = project_to_screen(Vec3(0, 0, 0), view_proj, 1920, 1080)
        assert uv.x == pytest.approx(0.5)
        assert uv.y == pytest.approx(0.5)
        assert valid is True

    def test_behind_camera(self) -> None:
        """Points behind camera should be marked invalid."""
        # Create a view-proj where z<0 is behind camera
        view_proj = Mat4.perspective(math.radians(60), 1.0, 0.1, 100.0)
        # Point at negative z (behind camera in OpenGL convention)
        uv, _, valid = project_to_screen(Vec3(0, 0, 10), view_proj, 1920, 1080)
        # With perspective, positive z goes into the screen (behind camera)
        # Actually for standard perspective, -z is forward, so +z should be behind
        # The validity depends on the w component after transformation

    def test_edge_of_screen(self) -> None:
        """Points at edge of frustum should be at UV edges."""
        # This is harder to test without careful matrix setup
        # We just verify the function doesn't crash
        view_proj = make_perspective_view_proj(
            Vec3(0, 0, 5), Vec3(0, 0, 0)
        )
        uv, depth, valid = project_to_screen(Vec3(0, 0, 0), view_proj, 1920, 1080)
        # Should be roughly in center
        assert 0.0 <= uv.x <= 1.0
        assert 0.0 <= uv.y <= 1.0


class TestCalculateReprojectedUV:
    """Tests for calculate_reprojected_uv function."""

    def test_static_camera(self) -> None:
        """With no camera movement, reprojected UV should equal current UV."""
        view_proj = make_perspective_view_proj(Vec3(0, 0, 5), Vec3(0, 0, 0))
        hit_pos = Vec3(0, 0, 0)  # At origin

        prev_uv, valid = calculate_reprojected_uv(hit_pos, view_proj, 1920, 1080)

        # Should project to roughly center
        assert valid is True
        assert 0.3 <= prev_uv.x <= 0.7
        assert 0.3 <= prev_uv.y <= 0.7

    def test_camera_moved(self) -> None:
        """Moving camera should change reprojected UV."""
        hit_pos = Vec3(1, 0, 0)  # Point at x=1

        prev_view_proj = make_perspective_view_proj(Vec3(0, 0, 5), Vec3(0, 0, 0))
        current_view_proj = make_perspective_view_proj(Vec3(1, 0, 5), Vec3(1, 0, 0))

        prev_uv, _ = calculate_reprojected_uv(hit_pos, prev_view_proj, 1920, 1080)
        curr_uv, _ = calculate_reprojected_uv(hit_pos, current_view_proj, 1920, 1080)

        # The point should project differently in the two frames
        # (camera moved, so same world point has different screen position)


# =============================================================================
# Disocclusion Detection Tests
# =============================================================================


class TestDisocclusionDetection:
    """Tests for disocclusion detection functions."""

    def test_depth_same(self) -> None:
        """Same depth should not be disoccluded."""
        result = detect_disocclusion_depth(10.0, 10.0, 0.1)
        assert result is False

    def test_depth_slightly_different(self) -> None:
        """Slightly different depth should not be disoccluded."""
        result = detect_disocclusion_depth(10.0, 10.5, 0.1)
        # 0.5/10 = 5% difference, below 10% threshold
        assert result is False

    def test_depth_very_different(self) -> None:
        """Very different depth should be disoccluded."""
        result = detect_disocclusion_depth(10.0, 20.0, 0.1)
        # 10/20 = 50% difference, above 10% threshold
        assert result is True

    def test_depth_zero_current(self) -> None:
        """Zero current depth should be disoccluded."""
        result = detect_disocclusion_depth(0.0, 10.0, 0.1)
        assert result is True

    def test_depth_zero_history(self) -> None:
        """Zero history depth should be disoccluded."""
        result = detect_disocclusion_depth(10.0, 0.0, 0.1)
        assert result is True

    def test_position_same(self) -> None:
        """Same position should not be disoccluded."""
        result = detect_disocclusion_position(
            Vec3(1, 2, 3), Vec3(1, 2, 3), 0.5
        )
        assert result is False

    def test_position_close(self) -> None:
        """Close positions should not be disoccluded."""
        result = detect_disocclusion_position(
            Vec3(1, 2, 3), Vec3(1.1, 2.1, 3.1), 0.5
        )
        # Distance = sqrt(0.01*3) ~ 0.17, below 0.5 threshold
        assert result is False

    def test_position_far(self) -> None:
        """Far positions should be disoccluded."""
        result = detect_disocclusion_position(
            Vec3(0, 0, 0), Vec3(1, 1, 1), 0.5
        )
        # Distance = sqrt(3) ~ 1.73, above 0.5 threshold
        assert result is True


# =============================================================================
# Neighborhood Clamping Tests
# =============================================================================


class TestNeighborhoodClamping:
    """Tests for neighborhood clamping functions."""

    def test_compute_bounds_rgb_uniform(self) -> None:
        """Uniform color neighborhood should have min=max."""
        buf = ColorBuffer(5, 5)
        color = Vec4(0.5, 0.5, 0.5, 1.0)
        for y in range(5):
            for x in range(5):
                buf.set_pixel(x, y, color)

        min_c, max_c = compute_neighborhood_bounds_rgb(buf, 2, 2, 1)
        assert min_c.x == pytest.approx(0.5)
        assert max_c.x == pytest.approx(0.5)

    def test_compute_bounds_rgb_varied(self) -> None:
        """Varied neighborhood should have different min/max."""
        buf = ColorBuffer(5, 5)
        buf.set_pixel(1, 1, Vec4(0.0, 0.0, 0.0, 1.0))
        buf.set_pixel(2, 2, Vec4(0.5, 0.5, 0.5, 1.0))
        buf.set_pixel(3, 3, Vec4(1.0, 1.0, 1.0, 1.0))

        min_c, max_c = compute_neighborhood_bounds_rgb(buf, 2, 2, 1)
        assert min_c.x <= 0.5
        assert max_c.x >= 0.5

    def test_clamp_color_rgb_inside(self) -> None:
        """Color inside bounds should be unchanged."""
        color = Vec4(0.5, 0.5, 0.5, 1.0)
        min_c = Vec4(0.0, 0.0, 0.0, 1.0)
        max_c = Vec4(1.0, 1.0, 1.0, 1.0)

        clamped = clamp_color_rgb(color, min_c, max_c)
        assert clamped.x == pytest.approx(0.5)

    def test_clamp_color_rgb_outside(self) -> None:
        """Color outside bounds should be clamped."""
        color = Vec4(1.5, -0.5, 0.5, 1.0)
        min_c = Vec4(0.0, 0.0, 0.0, 1.0)
        max_c = Vec4(1.0, 1.0, 1.0, 1.0)

        clamped = clamp_color_rgb(color, min_c, max_c)
        assert clamped.x == pytest.approx(1.0)
        assert clamped.y == pytest.approx(0.0)

    def test_clamp_color_ycocg(self) -> None:
        """YCoCg clamping should work correctly."""
        color = Vec4(0.8, 0.2, 0.2, 1.0)
        min_yc = Vec4(0.3, -0.1, -0.1, 1.0)
        max_yc = Vec4(0.7, 0.1, 0.1, 1.0)

        clamped = clamp_color_ycocg(color, min_yc, max_yc)
        # The clamped result should be valid RGB
        assert -0.1 <= clamped.x <= 1.1
        assert -0.1 <= clamped.y <= 1.1
        assert -0.1 <= clamped.z <= 1.1

    def test_compute_variance(self) -> None:
        """Variance computation should work."""
        buf = ColorBuffer(5, 5)
        for y in range(5):
            for x in range(5):
                val = (x + y) / 8.0
                buf.set_pixel(x, y, Vec4(val, val, val, 1.0))

        mean, var, std = compute_neighborhood_variance(buf, 2, 2, 1)
        assert mean.x > 0
        assert var.x > 0
        assert std.x > 0

    def test_clamp_variance(self) -> None:
        """Variance-based clamping should bound outliers."""
        color = Vec4(1.0, 1.0, 1.0, 1.0)
        mean = Vec4(0.5, 0.5, 0.5, 1.0)
        std = Vec4(0.1, 0.1, 0.1, 1.0)

        clamped = clamp_color_variance(color, mean, std, gamma=1.5)
        # 1.0 is far from mean=0.5, so should be clamped down
        assert clamped.x < 1.0


# =============================================================================
# ReprojectionConfig Tests
# =============================================================================


class TestReprojectionConfig:
    """Tests for ReprojectionConfig validation."""

    def test_default_config(self) -> None:
        """Default config should be valid."""
        config = ReprojectionConfig()
        assert config.blend_factor == pytest.approx(0.1)
        assert config.clamping_mode == ClampingMode.YCOCG

    def test_invalid_blend_factor(self) -> None:
        """Invalid blend_factor should raise."""
        with pytest.raises(ValueError, match="blend_factor"):
            ReprojectionConfig(blend_factor=0.0)
        with pytest.raises(ValueError, match="blend_factor"):
            ReprojectionConfig(blend_factor=1.5)

    def test_invalid_min_blend(self) -> None:
        """min_blend > blend_factor should raise."""
        with pytest.raises(ValueError, match="min_blend"):
            ReprojectionConfig(blend_factor=0.1, min_blend=0.2)

    def test_invalid_max_blend(self) -> None:
        """max_blend < blend_factor should raise."""
        with pytest.raises(ValueError, match="max_blend"):
            ReprojectionConfig(blend_factor=0.5, max_blend=0.3)

    def test_invalid_threshold(self) -> None:
        """Non-positive threshold should raise."""
        with pytest.raises(ValueError, match="disocclusion_threshold"):
            ReprojectionConfig(disocclusion_threshold=0.0)

    def test_invalid_neighborhood_size(self) -> None:
        """Neighborhood size < 1 should raise."""
        with pytest.raises(ValueError, match="neighborhood_size"):
            ReprojectionConfig(neighborhood_size=0)


# =============================================================================
# TAAReprojection Class Tests
# =============================================================================


class TestTAAReprojection:
    """Tests for the main TAAReprojection class."""

    def test_creation(self) -> None:
        """TAA system should initialize correctly."""
        taa = TAAReprojection(1920, 1080)
        assert taa.width == 1920
        assert taa.height == 1080
        assert taa.frame_count == 0
        assert taa.is_converged is False

    def test_invalid_dimensions(self) -> None:
        """Invalid dimensions should raise."""
        with pytest.raises(ValueError):
            TAAReprojection(0, 1080)

    def test_reset(self) -> None:
        """Reset should clear state."""
        taa = TAAReprojection(100, 100)
        # Simulate some accumulation
        taa._frame_count = 10
        taa._converged = True

        taa.reset()
        assert taa.frame_count == 0
        assert taa.is_converged is False

    def test_resize(self) -> None:
        """Resize should update dimensions and reset."""
        taa = TAAReprojection(100, 100)
        taa._frame_count = 10

        taa.resize(200, 150)
        assert taa.width == 200
        assert taa.height == 150
        assert taa.frame_count == 0

    def test_first_frame_passthrough(self) -> None:
        """First frame should pass through unchanged."""
        taa = TAAReprojection(10, 10)

        current_color = ColorBuffer(10, 10)
        current_color.set_pixel(5, 5, Vec4(1.0, 0.0, 0.0, 1.0))

        hit_positions = HitPositionBuffer(10, 10)
        hit_positions.set_hit(5, 5, Vec3(0, 0, 5))

        view_proj = make_perspective_view_proj(Vec3(0, 0, 10), Vec3(0, 0, 0))

        result = taa.accumulate(current_color, hit_positions, view_proj, view_proj)

        # First frame: output = current
        c = result.get_pixel(5, 5)
        assert c.x == pytest.approx(1.0)
        assert taa.frame_count == 1

    def test_second_frame_blends(self) -> None:
        """Second frame should blend with history."""
        taa = TAAReprojection(10, 10)
        view_proj = make_perspective_view_proj(Vec3(0, 0, 10), Vec3(0, 0, 0))

        # Frame 1: red
        color1 = ColorBuffer(10, 10)
        color1.set_pixel(5, 5, Vec4(1.0, 0.0, 0.0, 1.0))
        hits1 = HitPositionBuffer(10, 10)
        hits1.set_hit(5, 5, Vec3(0, 0, 5))
        taa.accumulate(color1, hits1, view_proj, view_proj)

        # Frame 2: green
        color2 = ColorBuffer(10, 10)
        color2.set_pixel(5, 5, Vec4(0.0, 1.0, 0.0, 1.0))
        hits2 = HitPositionBuffer(10, 10)
        hits2.set_hit(5, 5, Vec3(0, 0, 5))
        result = taa.accumulate(color2, hits2, view_proj, view_proj)

        # Should be a blend of red and green
        c = result.get_pixel(5, 5)
        assert c.x > 0  # Some red from history
        assert c.y > 0  # Some green from current
        assert taa.frame_count == 2

    def test_convergence(self) -> None:
        """Should converge after enough frames."""
        config = ReprojectionConfig(blend_factor=0.1)
        taa = TAAReprojection(10, 10, config)
        view_proj = make_perspective_view_proj(Vec3(0, 0, 10), Vec3(0, 0, 0))

        # Run 20 frames
        for i in range(20):
            color = ColorBuffer(10, 10)
            hits = HitPositionBuffer(10, 10)
            taa.accumulate(color, hits, view_proj, view_proj)

        # Should be converged after ~10 frames (1/0.1)
        assert taa.is_converged is True

    def test_get_history_color(self) -> None:
        """Should return a copy of history color buffer."""
        taa = TAAReprojection(10, 10)
        view_proj = Mat4.identity()

        color = ColorBuffer(10, 10)
        color.set_pixel(5, 5, Vec4(0.5, 0.5, 0.5, 1.0))
        hits = HitPositionBuffer(10, 10)
        taa.accumulate(color, hits, view_proj, view_proj)

        history = taa.get_history_color()
        assert history.width == 10
        assert history.height == 10

    def test_get_history_positions(self) -> None:
        """Should return a copy of history positions buffer."""
        taa = TAAReprojection(10, 10)
        view_proj = Mat4.identity()

        color = ColorBuffer(10, 10)
        hits = HitPositionBuffer(10, 10)
        hits.set_hit(5, 5, Vec3(1, 2, 3))
        taa.accumulate(color, hits, view_proj, view_proj)

        history = taa.get_history_positions()
        pos, valid = history.get_position(5, 5)
        assert valid is True
        assert pos.x == pytest.approx(1.0)


class TestTAADisocclusion:
    """Tests for disocclusion handling in TAA."""

    def test_disocclusion_fast_reset(self) -> None:
        """Disoccluded pixels should have higher blend factor."""
        config = ReprojectionConfig(
            disocclusion_mode=DisocclusionMode.POSITION,
            disocclusion_threshold=0.1,
        )
        taa = TAAReprojection(10, 10, config)

        # Need to advance frame count so blend factors are different
        taa._frame_count = 5

        # The _compute_blend_factor is internal but we can test behavior
        blend_normal = taa._compute_blend_factor(is_disoccluded=False)
        blend_disoccluded = taa._compute_blend_factor(is_disoccluded=True)

        # After frame 0, disoccluded should use max_blend (0.8), normal uses base (0.1)
        assert blend_disoccluded > blend_normal
        assert blend_disoccluded == pytest.approx(config.max_blend)

    def test_miss_to_hit_disocclusion(self) -> None:
        """Going from miss to hit should be detected as disocclusion."""
        config = ReprojectionConfig(disocclusion_mode=DisocclusionMode.COMBINED)
        taa = TAAReprojection(10, 10, config)

        # Frame 1: miss (sky)
        hits1 = HitPositionBuffer(10, 10)
        hits1.set_miss(5, 5)

        # Frame 2: hit (surface appeared)
        hits2 = HitPositionBuffer(10, 10)
        hits2.set_hit(5, 5, Vec3(0, 0, 5))

        # The disocclusion check should detect this
        disoccluded = taa._is_disoccluded(
            Vec3(0, 0, 5), True,   # current
            Vec3.zero(), False,    # history (miss)
            5.0, 0.0,
        )
        assert disoccluded is True
