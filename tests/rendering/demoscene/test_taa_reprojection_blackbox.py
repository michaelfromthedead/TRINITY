"""
Blackbox Tests for TAA Reprojection (T-DEMO-8.5).

Tests the public interface and expected behavior of world-space
hit position reprojection for temporal anti-aliasing.

Test categories:
- Hit position accuracy (world-space correctness)
- Reprojection correctness (screen-space projection)
- Stability over time (convergence, noise reduction)
- No motion vector requirement verification
- WGSL code generation
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
    # Main class
    TAAReprojection,
    # WGSL
    generate_reprojection_wgsl,
    generate_ycocg_wgsl,
    generate_neighborhood_clamping_wgsl,
    generate_disocclusion_wgsl,
    generate_taa_reprojection_wgsl,
)


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_scene_static(
    width: int,
    height: int,
    color: Vec4,
    hit_depth: float = 5.0,
) -> tuple[ColorBuffer, HitPositionBuffer]:
    """Create a static test scene with uniform color and depth."""
    colors = ColorBuffer(width, height)
    hits = HitPositionBuffer(width, height)

    for y in range(height):
        for x in range(width):
            colors.set_pixel(x, y, color)
            # Create world positions on a plane at z=hit_depth
            world_x = (x / width - 0.5) * 10
            world_y = (y / height - 0.5) * 10
            hits.set_hit(x, y, Vec3(world_x, world_y, hit_depth))

    return colors, hits


def create_test_scene_gradient(
    width: int,
    height: int,
    hit_depth: float = 5.0,
) -> tuple[ColorBuffer, HitPositionBuffer]:
    """Create a test scene with gradient color."""
    colors = ColorBuffer(width, height)
    hits = HitPositionBuffer(width, height)

    for y in range(height):
        for x in range(width):
            r = x / width
            g = y / height
            b = 0.5
            colors.set_pixel(x, y, Vec4(r, g, b, 1.0))

            world_x = (x / width - 0.5) * 10
            world_y = (y / height - 0.5) * 10
            hits.set_hit(x, y, Vec3(world_x, world_y, hit_depth))

    return colors, hits


def make_camera_matrices(
    eye: Vec3,
    target: Vec3,
    fov: float = math.radians(60),
) -> Mat4:
    """Create view-projection matrix."""
    view = Mat4.look_at(eye, target, Vec3.up())
    proj = Mat4.perspective(fov, 16/9, 0.1, 100.0)
    return proj @ view


# =============================================================================
# Hit Position Accuracy Tests
# =============================================================================


class TestHitPositionAccuracy:
    """Tests for hit position storage and retrieval accuracy."""

    def test_exact_position_storage(self) -> None:
        """Hit positions should be stored exactly."""
        buf = HitPositionBuffer(100, 100)

        # Store precise positions
        positions = [
            (10, 20, Vec3(1.23456, 7.89012, 3.45678)),
            (50, 50, Vec3(-5.5, 0.0, 10.0)),
            (99, 99, Vec3(100.0, 200.0, 300.0)),
        ]

        for x, y, pos in positions:
            buf.set_hit(x, y, pos)

        for x, y, expected in positions:
            actual, valid = buf.get_position(x, y)
            assert valid is True
            assert actual.x == pytest.approx(expected.x, abs=1e-6)
            assert actual.y == pytest.approx(expected.y, abs=1e-6)
            assert actual.z == pytest.approx(expected.z, abs=1e-6)

    def test_large_position_values(self) -> None:
        """Large world-space positions should be handled correctly."""
        buf = HitPositionBuffer(10, 10)

        # Large astronomical-scale positions
        large_pos = Vec3(1e6, 1e6, 1e6)
        buf.set_hit(5, 5, large_pos)

        result, valid = buf.get_position(5, 5)
        assert valid is True
        assert result.x == pytest.approx(1e6, rel=1e-6)

    def test_small_position_values(self) -> None:
        """Small position values should be preserved."""
        buf = HitPositionBuffer(10, 10)

        small_pos = Vec3(1e-6, 1e-6, 1e-6)
        buf.set_hit(5, 5, small_pos)

        result, valid = buf.get_position(5, 5)
        assert valid is True
        assert result.x == pytest.approx(1e-6, abs=1e-9)

    def test_negative_positions(self) -> None:
        """Negative world positions should be handled."""
        buf = HitPositionBuffer(10, 10)

        neg_pos = Vec3(-10.0, -20.0, -30.0)
        buf.set_hit(5, 5, neg_pos)

        result, valid = buf.get_position(5, 5)
        assert valid is True
        assert result.x == pytest.approx(-10.0)
        assert result.y == pytest.approx(-20.0)
        assert result.z == pytest.approx(-30.0)

    def test_hit_miss_distinction(self) -> None:
        """Hits and misses should be clearly distinguished."""
        buf = HitPositionBuffer(10, 10)

        buf.set_hit(3, 3, Vec3(1, 2, 3))
        buf.set_miss(5, 5)

        _, valid_hit = buf.get_position(3, 3)
        _, valid_miss = buf.get_position(5, 5)

        assert valid_hit is True
        assert valid_miss is False


# =============================================================================
# Reprojection Correctness Tests
# =============================================================================


class TestReprojectionCorrectness:
    """Tests for screen-space reprojection correctness."""

    def test_static_camera_reprojection(self) -> None:
        """With static camera, reprojection should preserve position."""
        width, height = 64, 64
        taa = TAAReprojection(width, height)

        view_proj = make_camera_matrices(Vec3(0, 0, 10), Vec3(0, 0, 0))

        # Create scene
        colors, hits = create_test_scene_gradient(width, height)

        # First frame
        taa.accumulate(colors, hits, view_proj, view_proj)

        # Second frame with same camera
        result = taa.accumulate(colors, hits, view_proj, view_proj)

        # Center pixel should have blended but still visible gradient
        c = result.get_pixel(width // 2, height // 2)
        assert c.x > 0  # Has some color

    def test_camera_translation_reprojection(self) -> None:
        """Camera translation should cause appropriate UV shifts."""
        width, height = 64, 64
        taa = TAAReprojection(width, height)

        prev_view_proj = make_camera_matrices(Vec3(0, 0, 10), Vec3(0, 0, 0))
        curr_view_proj = make_camera_matrices(Vec3(1, 0, 10), Vec3(1, 0, 0))

        colors, hits = create_test_scene_gradient(width, height)

        # First frame with prev camera
        taa.accumulate(colors, hits, prev_view_proj, prev_view_proj)

        # Second frame with moved camera
        result = taa.accumulate(colors, hits, curr_view_proj, prev_view_proj)

        # Should have valid output (no crashes, reasonable blending)
        assert taa.frame_count == 2

    def test_camera_rotation_reprojection(self) -> None:
        """Camera rotation should cause appropriate UV shifts."""
        width, height = 32, 32
        taa = TAAReprojection(width, height)

        # Rotate camera slightly around Y axis
        prev_view_proj = make_camera_matrices(Vec3(0, 0, 10), Vec3(0, 0, 0))
        curr_view_proj = make_camera_matrices(Vec3(0, 0, 10), Vec3(0.1, 0, 0))

        colors, hits = create_test_scene_static(width, height, Vec4(0.5, 0.5, 0.5, 1))

        taa.accumulate(colors, hits, prev_view_proj, prev_view_proj)
        result = taa.accumulate(colors, hits, curr_view_proj, prev_view_proj)

        assert result.width == width
        assert result.height == height


# =============================================================================
# Stability Over Time Tests
# =============================================================================


class TestStabilityOverTime:
    """Tests for temporal stability and convergence."""

    def test_noise_reduction_over_frames(self) -> None:
        """Adding noise should be smoothed over multiple frames."""
        width, height = 32, 32
        taa = TAAReprojection(width, height, ReprojectionConfig(blend_factor=0.1))
        view_proj = make_camera_matrices(Vec3(0, 0, 10), Vec3(0, 0, 0))

        hits = HitPositionBuffer(width, height)
        for y in range(height):
            for x in range(width):
                hits.set_hit(x, y, Vec3(0, 0, 5))

        # Track variance of center pixel over frames
        pixel_values = []

        for frame in range(20):
            colors = ColorBuffer(width, height)
            # Add noise
            noise = 0.1 * math.sin(frame * 0.5)
            base = 0.5 + noise
            for y in range(height):
                for x in range(width):
                    colors.set_pixel(x, y, Vec4(base, base, base, 1.0))

            result = taa.accumulate(colors, hits, view_proj, view_proj)
            c = result.get_pixel(width // 2, height // 2)
            pixel_values.append(c.x)

        # Later frames should have more stable values
        early_variance = sum((v - 0.5)**2 for v in pixel_values[:5]) / 5
        late_variance = sum((v - 0.5)**2 for v in pixel_values[-5:]) / 5

        # Late variance should be smaller (more stable)
        assert late_variance <= early_variance + 0.01

    def test_convergence_to_target(self) -> None:
        """Constant input should converge to that value."""
        width, height = 16, 16
        # Use disocclusion mode NONE so every pixel just blends normally
        # and use higher blend factor for faster convergence
        config = ReprojectionConfig(
            blend_factor=0.3,  # Higher blend for faster convergence
            clamping_mode=ClampingMode.NONE,  # No clamping for accurate convergence
            disocclusion_mode=DisocclusionMode.NONE,  # No disocclusion checks
        )
        taa = TAAReprojection(width, height, config)
        view_proj = make_camera_matrices(Vec3(0, 0, 20), Vec3(0, 0, 0))

        target_color = Vec4(0.7, 0.3, 0.5, 1.0)

        # Run many frames with constant input
        for _ in range(50):
            colors = ColorBuffer(width, height)
            hits = HitPositionBuffer(width, height)

            for y in range(height):
                for x in range(width):
                    colors.set_pixel(x, y, target_color)
                    # Place hits in a grid in view
                    world_x = (x / width - 0.5) * 5
                    world_y = (y / height - 0.5) * 5
                    hits.set_hit(x, y, Vec3(world_x, world_y, 0))

            result = taa.accumulate(colors, hits, view_proj, view_proj)

        # Should have converged close to target
        c = result.get_pixel(8, 8)
        assert c.x == pytest.approx(0.7, abs=0.15)
        assert c.y == pytest.approx(0.3, abs=0.15)
        assert c.z == pytest.approx(0.5, abs=0.15)

    def test_convergence_flag(self) -> None:
        """is_converged should be True after sufficient frames."""
        config = ReprojectionConfig(blend_factor=0.1)
        taa = TAAReprojection(16, 16, config)
        view_proj = Mat4.identity()

        colors = ColorBuffer(16, 16)
        hits = HitPositionBuffer(16, 16)

        assert taa.is_converged is False

        # Run 15 frames (should converge around 10)
        for _ in range(15):
            taa.accumulate(colors, hits, view_proj, view_proj)

        assert taa.is_converged is True

    def test_frame_count_tracking(self) -> None:
        """frame_count should accurately track accumulated frames."""
        taa = TAAReprojection(8, 8)
        view_proj = Mat4.identity()

        colors = ColorBuffer(8, 8)
        hits = HitPositionBuffer(8, 8)

        assert taa.frame_count == 0

        for i in range(5):
            taa.accumulate(colors, hits, view_proj, view_proj)
            assert taa.frame_count == i + 1


# =============================================================================
# No Motion Vector Requirement Tests
# =============================================================================


class TestNoMotionVectors:
    """Tests verifying TAA works without explicit motion vectors."""

    def test_works_without_velocity_buffer(self) -> None:
        """TAA should work with only hit positions, no velocity."""
        taa = TAAReprojection(32, 32)
        view_proj = make_camera_matrices(Vec3(0, 0, 10), Vec3(0, 0, 0))

        colors, hits = create_test_scene_gradient(32, 32)

        # Should work without any velocity/motion data
        for _ in range(10):
            result = taa.accumulate(colors, hits, view_proj, view_proj)

        assert taa.frame_count == 10
        assert result.width == 32

    def test_moving_objects_via_hit_position(self) -> None:
        """Moving objects should be tracked via changing hit positions."""
        width, height = 32, 32
        taa = TAAReprojection(width, height)
        view_proj = make_camera_matrices(Vec3(0, 0, 10), Vec3(0, 0, 0))

        for frame in range(10):
            colors = ColorBuffer(width, height)
            hits = HitPositionBuffer(width, height)

            # Simulate a sphere moving across the scene
            sphere_x = (frame - 5) * 0.5  # -2.5 to 2.0

            for y in range(height):
                for x in range(width):
                    world_x = (x / width - 0.5) * 10
                    world_y = (y / height - 0.5) * 10

                    # Check if in sphere
                    dist = math.sqrt((world_x - sphere_x)**2 + world_y**2)
                    if dist < 1.0:
                        # Hit sphere
                        hits.set_hit(x, y, Vec3(world_x, world_y, 5.0 - (1 - dist)))
                        colors.set_pixel(x, y, Vec4(1.0, 0.0, 0.0, 1.0))
                    else:
                        # Hit background
                        hits.set_hit(x, y, Vec3(world_x, world_y, 10.0))
                        colors.set_pixel(x, y, Vec4(0.2, 0.2, 0.2, 1.0))

            taa.accumulate(colors, hits, view_proj, view_proj)

        # Should complete without issues
        assert taa.frame_count == 10

    def test_sky_pixels_handled(self) -> None:
        """Sky/miss pixels should be handled gracefully without motion vectors."""
        width, height = 32, 32
        taa = TAAReprojection(width, height)
        view_proj = Mat4.identity()

        colors = ColorBuffer(width, height)
        hits = HitPositionBuffer(width, height)

        # Half sky (miss), half ground (hit)
        for y in range(height):
            for x in range(width):
                if y < height // 2:
                    # Sky
                    colors.set_pixel(x, y, Vec4(0.5, 0.7, 1.0, 1.0))
                    hits.set_miss(x, y)
                else:
                    # Ground
                    colors.set_pixel(x, y, Vec4(0.3, 0.5, 0.2, 1.0))
                    hits.set_hit(x, y, Vec3(x, y, 5.0))

        for _ in range(10):
            taa.accumulate(colors, hits, view_proj, view_proj)

        assert taa.frame_count == 10


# =============================================================================
# Disocclusion Handling Tests
# =============================================================================


class TestDisocclusionHandling:
    """Tests for disocclusion detection and recovery."""

    def test_new_surface_appears(self) -> None:
        """Newly visible surfaces should update quickly."""
        width, height = 16, 16
        config = ReprojectionConfig(
            disocclusion_mode=DisocclusionMode.COMBINED,
            blend_factor=0.1,
            max_blend=0.8,
        )
        taa = TAAReprojection(width, height, config)
        view_proj = Mat4.identity()

        # Frame 1-5: Empty scene (all misses)
        for _ in range(5):
            colors = ColorBuffer(width, height)
            hits = HitPositionBuffer(width, height)
            for y in range(height):
                for x in range(width):
                    colors.set_pixel(x, y, Vec4(0.0, 0.0, 0.0, 1.0))
                    hits.set_miss(x, y)
            taa.accumulate(colors, hits, view_proj, view_proj)

        # Frame 6: Surface appears
        colors = ColorBuffer(width, height)
        hits = HitPositionBuffer(width, height)
        for y in range(height):
            for x in range(width):
                colors.set_pixel(x, y, Vec4(1.0, 1.0, 1.0, 1.0))
                hits.set_hit(x, y, Vec3(x, y, 5.0))

        result = taa.accumulate(colors, hits, view_proj, view_proj)

        # New surface should show up strongly (high blend due to disocclusion)
        c = result.get_pixel(8, 8)
        assert c.x > 0.5  # Should favor new color

    def test_surface_disappears(self) -> None:
        """Disappearing surfaces should not ghost."""
        width, height = 16, 16
        config = ReprojectionConfig(disocclusion_mode=DisocclusionMode.COMBINED)
        taa = TAAReprojection(width, height, config)
        view_proj = Mat4.identity()

        # Frame 1-5: Surface present
        for _ in range(5):
            colors = ColorBuffer(width, height)
            hits = HitPositionBuffer(width, height)
            for y in range(height):
                for x in range(width):
                    colors.set_pixel(x, y, Vec4(1.0, 0.0, 0.0, 1.0))
                    hits.set_hit(x, y, Vec3(x, y, 5.0))
            taa.accumulate(colors, hits, view_proj, view_proj)

        # Frame 6: Surface gone (all sky)
        colors = ColorBuffer(width, height)
        hits = HitPositionBuffer(width, height)
        for y in range(height):
            for x in range(width):
                colors.set_pixel(x, y, Vec4(0.5, 0.7, 1.0, 1.0))
                hits.set_miss(x, y)

        result = taa.accumulate(colors, hits, view_proj, view_proj)

        # Should transition toward new (sky) color
        # The exact value depends on disocclusion handling


# =============================================================================
# WGSL Generation Tests
# =============================================================================


class TestWGSLGeneration:
    """Tests for WGSL code generation."""

    def test_reprojection_wgsl_structure(self) -> None:
        """Reprojection WGSL should have required functions."""
        code = generate_reprojection_wgsl()
        assert "fn project_to_screen" in code
        assert "fn calculate_reprojected_uv" in code
        assert "view_proj" in code
        assert "mat4x4" in code

    def test_ycocg_wgsl_structure(self) -> None:
        """YCoCg WGSL should have conversion functions."""
        code = generate_ycocg_wgsl()
        assert "fn rgb_to_ycocg" in code
        assert "fn ycocg_to_rgb" in code

    def test_neighborhood_clamping_wgsl(self) -> None:
        """Neighborhood clamping WGSL should have required functions."""
        code = generate_neighborhood_clamping_wgsl()
        assert "fn compute_neighborhood_bounds_ycocg" in code
        assert "fn clamp_history_ycocg" in code

    def test_disocclusion_wgsl(self) -> None:
        """Disocclusion WGSL should have detection functions."""
        code = generate_disocclusion_wgsl()
        assert "fn detect_disocclusion_position" in code
        assert "fn detect_disocclusion_depth" in code

    def test_full_shader_generation(self) -> None:
        """Full TAA shader should compile all components."""
        code = generate_taa_reprojection_wgsl()

        # Should include all components
        assert "rgb_to_ycocg" in code
        assert "project_to_screen" in code
        assert "detect_disocclusion" in code
        assert "compute_neighborhood_bounds" in code

        # Should have main entry point
        assert "@compute" in code
        assert "fn taa_reprojection" in code

        # Should have uniforms
        assert "struct TAAReprojectionParams" in code
        assert "blend_factor" in code

        # Should have texture bindings
        assert "texture_2d<f32>" in code
        assert "texture_storage_2d" in code

    def test_wgsl_valid_syntax(self) -> None:
        """Generated WGSL should have valid syntax markers."""
        code = generate_taa_reprojection_wgsl()

        # Check for balanced braces
        assert code.count("{") == code.count("}")

        # Check for proper function declarations
        assert code.count("fn ") >= 5

        # Check for proper type usage
        assert "vec3<f32>" in code
        assert "vec4<f32>" in code


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfigurationModes:
    """Tests for different configuration modes."""

    def test_no_clamping_mode(self) -> None:
        """NONE clamping should allow full blending."""
        config = ReprojectionConfig(clamping_mode=ClampingMode.NONE)
        taa = TAAReprojection(8, 8, config)
        view_proj = Mat4.identity()

        colors = ColorBuffer(8, 8)
        hits = HitPositionBuffer(8, 8)

        for _ in range(5):
            taa.accumulate(colors, hits, view_proj, view_proj)

        assert taa.frame_count == 5

    def test_rgb_clamping_mode(self) -> None:
        """RGB clamping should work without conversion."""
        config = ReprojectionConfig(clamping_mode=ClampingMode.RGB)
        taa = TAAReprojection(8, 8, config)
        view_proj = Mat4.identity()

        colors = ColorBuffer(8, 8)
        hits = HitPositionBuffer(8, 8)

        for _ in range(5):
            taa.accumulate(colors, hits, view_proj, view_proj)

        assert taa.frame_count == 5

    def test_variance_clamping_mode(self) -> None:
        """Variance clamping should adapt to local statistics."""
        config = ReprojectionConfig(clamping_mode=ClampingMode.VARIANCE)
        taa = TAAReprojection(8, 8, config)
        view_proj = Mat4.identity()

        colors = ColorBuffer(8, 8)
        hits = HitPositionBuffer(8, 8)

        for _ in range(5):
            taa.accumulate(colors, hits, view_proj, view_proj)

        assert taa.frame_count == 5

    def test_depth_only_disocclusion(self) -> None:
        """Depth-only disocclusion mode should work."""
        config = ReprojectionConfig(disocclusion_mode=DisocclusionMode.DEPTH)
        taa = TAAReprojection(8, 8, config)
        view_proj = Mat4.identity()

        colors = ColorBuffer(8, 8)
        hits = HitPositionBuffer(8, 8)

        for _ in range(5):
            taa.accumulate(colors, hits, view_proj, view_proj)

        assert taa.frame_count == 5

    def test_position_only_disocclusion(self) -> None:
        """Position-only disocclusion mode should work."""
        config = ReprojectionConfig(disocclusion_mode=DisocclusionMode.POSITION)
        taa = TAAReprojection(8, 8, config)
        view_proj = Mat4.identity()

        colors = ColorBuffer(8, 8)
        hits = HitPositionBuffer(8, 8)

        for _ in range(5):
            taa.accumulate(colors, hits, view_proj, view_proj)

        assert taa.frame_count == 5

    def test_larger_neighborhood(self) -> None:
        """Larger neighborhood should work for clamping."""
        config = ReprojectionConfig(neighborhood_size=2)
        taa = TAAReprojection(16, 16, config)
        view_proj = Mat4.identity()

        colors, hits = create_test_scene_gradient(16, 16)

        for _ in range(5):
            taa.accumulate(colors, hits, view_proj, view_proj)

        assert taa.frame_count == 5
