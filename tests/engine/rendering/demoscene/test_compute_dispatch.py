"""
Tests for compute_dispatch.py (T-DEMO-3.9).

Tests full-screen compute shader dispatch including:
  - Dispatch dimension calculation (ceiling division)
  - UV coordinate mapping for pixel centers
  - Bounds checking for partial workgroups
  - WGSL code generation

Acceptance Criteria:
  - Single dispatch covers entire viewport
  - Out-of-bounds pixels correctly skipped
  - UV coordinates correctly computed for pixel centers
  - 20+ tests covering dispatch dimensions, UV mapping
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.compute_dispatch import (
    ComputeDispatch,
    BindGroupConfig,
    OutputFormat,
    generate_entry_point_template,
    calculate_dispatch_dimensions,
)
from engine.rendering.demoscene.ast_nodes import RenderSettingsNode


# =============================================================================
# Dispatch Dimension Tests
# =============================================================================


class TestDispatchDimensions:
    """Tests for dispatch dimension calculation."""

    def test_exact_fit_1920x1080(self):
        """1920x1080 with 8x8 workgroups should have exact fit."""
        dispatch = ComputeDispatch(1920, 1080)
        x, y = dispatch.dispatch_dimensions()
        assert x == 240  # 1920 / 8 = 240
        assert y == 135  # 1080 / 8 = 135

    def test_exact_fit_small(self):
        """Small resolution with exact workgroup fit."""
        dispatch = ComputeDispatch(64, 64)
        x, y = dispatch.dispatch_dimensions()
        assert x == 8
        assert y == 8

    def test_partial_workgroup_x(self):
        """Non-divisible width should ceil up."""
        dispatch = ComputeDispatch(1001, 1080)
        x, y = dispatch.dispatch_dimensions()
        assert x == 126  # ceil(1001/8) = 126
        assert y == 135

    def test_partial_workgroup_y(self):
        """Non-divisible height should ceil up."""
        dispatch = ComputeDispatch(1920, 1081)
        x, y = dispatch.dispatch_dimensions()
        assert x == 240
        assert y == 136  # ceil(1081/8) = 136

    def test_partial_workgroup_both(self):
        """Non-divisible width and height."""
        dispatch = ComputeDispatch(1001, 601)
        x, y = dispatch.dispatch_dimensions()
        assert x == 126
        assert y == 76

    def test_single_pixel(self):
        """Single pixel resolution."""
        dispatch = ComputeDispatch(1, 1)
        x, y = dispatch.dispatch_dimensions()
        assert x == 1
        assert y == 1

    def test_workgroup_size_matches(self):
        """Resolution matching workgroup size."""
        dispatch = ComputeDispatch(8, 8)
        x, y = dispatch.dispatch_dimensions()
        assert x == 1
        assert y == 1

    def test_custom_workgroup_size(self):
        """Custom workgroup sizes."""
        dispatch = ComputeDispatch(100, 100, workgroup_size_x=16, workgroup_size_y=16)
        x, y = dispatch.dispatch_dimensions()
        assert x == 7  # ceil(100/16) = 7
        assert y == 7

    def test_asymmetric_workgroup(self):
        """Asymmetric workgroup size."""
        dispatch = ComputeDispatch(100, 100, workgroup_size_x=4, workgroup_size_y=8)
        x, y = dispatch.dispatch_dimensions()
        assert x == 25  # ceil(100/4)
        assert y == 13  # ceil(100/8)

    def test_4k_resolution(self):
        """4K resolution dispatch."""
        dispatch = ComputeDispatch(3840, 2160)
        x, y = dispatch.dispatch_dimensions()
        assert x == 480  # 3840 / 8
        assert y == 270  # 2160 / 8


class TestDispatchDimensionsConvenience:
    """Tests for standalone calculate_dispatch_dimensions function."""

    def test_basic_calculation(self):
        """Basic dispatch calculation."""
        x, y = calculate_dispatch_dimensions(1920, 1080)
        assert x == 240
        assert y == 135

    def test_with_custom_workgroup(self):
        """Calculation with custom workgroup size."""
        x, y = calculate_dispatch_dimensions(1000, 1000, 16, 16)
        assert x == 63  # ceil(1000/16)
        assert y == 63


# =============================================================================
# UV Coordinate Tests
# =============================================================================


class TestUVCoordinates:
    """Tests for UV coordinate mapping."""

    def test_pixel_center_uv_center(self):
        """Center pixel should map to near (0, 0)."""
        dispatch = ComputeDispatch(100, 100)
        u, v = dispatch.pixel_to_uv(49, 49)
        # Pixel 49 center = 49.5, normalized = 49.5/100 = 0.495
        # UV = 0.495 * 2 - 1 = -0.01
        assert abs(u - (-0.01)) < 0.001
        assert abs(v - 0.01) < 0.001

    def test_pixel_center_top_left(self):
        """Top-left pixel (0,0) should map to near (-1, 1)."""
        dispatch = ComputeDispatch(100, 100)
        u, v = dispatch.pixel_to_uv(0, 0)
        # x: (0 + 0.5)/100 * 2 - 1 = 0.01 - 1 = -0.99
        # y: 1 - (0 + 0.5)/100 * 2 = 1 - 0.01 = 0.99
        assert abs(u - (-0.99)) < 0.001
        assert abs(v - 0.99) < 0.001

    def test_pixel_center_bottom_right(self):
        """Bottom-right pixel should map to near (1, -1)."""
        dispatch = ComputeDispatch(100, 100)
        u, v = dispatch.pixel_to_uv(99, 99)
        # x: (99 + 0.5)/100 * 2 - 1 = 1.99 - 1 = 0.99
        # y: 1 - (99 + 0.5)/100 * 2 = 1 - 1.99 = -0.99
        assert abs(u - 0.99) < 0.001
        assert abs(v - (-0.99)) < 0.001

    def test_uv_range(self):
        """All UV coordinates should be in [-1, 1]."""
        dispatch = ComputeDispatch(1920, 1080)
        for x in [0, 959, 1919]:
            for y in [0, 539, 1079]:
                u, v = dispatch.pixel_to_uv(x, y)
                assert -1.0 <= u <= 1.0, f"u={u} out of range at ({x},{y})"
                assert -1.0 <= v <= 1.0, f"v={v} out of range at ({x},{y})"

    def test_uv_to_pixel_roundtrip(self):
        """uv_to_pixel should approximately invert pixel_to_uv."""
        dispatch = ComputeDispatch(100, 100)
        for px in [0, 25, 50, 75, 99]:
            for py in [0, 25, 50, 75, 99]:
                u, v = dispatch.pixel_to_uv(px, py)
                rx, ry = dispatch.uv_to_pixel(u, v)
                # May be off by 1 due to rounding
                assert abs(rx - px) <= 1, f"x mismatch: {rx} vs {px}"
                assert abs(ry - py) <= 1, f"y mismatch: {ry} vs {py}"

    def test_uv_symmetry(self):
        """UV coordinates should be symmetric around center."""
        dispatch = ComputeDispatch(100, 100)
        u1, v1 = dispatch.pixel_to_uv(25, 25)
        u2, v2 = dispatch.pixel_to_uv(74, 74)
        # These should be roughly symmetric
        assert abs(u1 + u2) < 0.05  # Should sum to ~0
        assert abs(v1 + v2) < 0.05


# =============================================================================
# Bounds Checking Tests
# =============================================================================


class TestBoundsChecking:
    """Tests for bounds checking."""

    def test_in_bounds_valid(self):
        """Valid coordinates should be in bounds."""
        dispatch = ComputeDispatch(100, 100)
        assert dispatch.is_in_bounds(0, 0)
        assert dispatch.is_in_bounds(50, 50)
        assert dispatch.is_in_bounds(99, 99)

    def test_in_bounds_edge(self):
        """Edge pixels should be in bounds."""
        dispatch = ComputeDispatch(100, 100)
        assert dispatch.is_in_bounds(0, 99)
        assert dispatch.is_in_bounds(99, 0)

    def test_out_of_bounds_negative(self):
        """Negative coordinates should be out of bounds."""
        dispatch = ComputeDispatch(100, 100)
        assert not dispatch.is_in_bounds(-1, 0)
        assert not dispatch.is_in_bounds(0, -1)

    def test_out_of_bounds_too_large(self):
        """Coordinates >= size should be out of bounds."""
        dispatch = ComputeDispatch(100, 100)
        assert not dispatch.is_in_bounds(100, 0)
        assert not dispatch.is_in_bounds(0, 100)
        assert not dispatch.is_in_bounds(100, 100)


class TestTotalThreads:
    """Tests for total thread calculation."""

    def test_total_threads_exact(self):
        """Total threads for exact fit."""
        dispatch = ComputeDispatch(64, 64)
        assert dispatch.total_threads() == 64 * 64

    def test_total_threads_partial(self):
        """Total threads includes partial workgroups."""
        dispatch = ComputeDispatch(65, 65)
        # dispatch is 9x9, so 9*8 * 9*8 = 72*72 = 5184
        dim_x, dim_y = dispatch.dispatch_dimensions()
        expected = dim_x * 8 * dim_y * 8
        assert dispatch.total_threads() == expected


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for parameter validation."""

    def test_valid_parameters(self):
        """Valid parameters should not raise."""
        dispatch = ComputeDispatch(1920, 1080, 8, 8)
        assert dispatch.width == 1920
        assert dispatch.height == 1080

    def test_invalid_width_zero(self):
        """Zero width should raise."""
        with pytest.raises(ValueError, match="Width must be positive"):
            ComputeDispatch(0, 100)

    def test_invalid_width_negative(self):
        """Negative width should raise."""
        with pytest.raises(ValueError, match="Width must be positive"):
            ComputeDispatch(-1, 100)

    def test_invalid_height_zero(self):
        """Zero height should raise."""
        with pytest.raises(ValueError, match="Height must be positive"):
            ComputeDispatch(100, 0)

    def test_invalid_workgroup_x(self):
        """Invalid workgroup X should raise."""
        with pytest.raises(ValueError, match="Workgroup X must be positive"):
            ComputeDispatch(100, 100, workgroup_size_x=0)

    def test_invalid_workgroup_y(self):
        """Invalid workgroup Y should raise."""
        with pytest.raises(ValueError, match="Workgroup Y must be positive"):
            ComputeDispatch(100, 100, workgroup_size_y=0)

    def test_workgroup_too_large(self):
        """Workgroup > 32 should raise."""
        with pytest.raises(ValueError, match="should not exceed 32"):
            ComputeDispatch(100, 100, workgroup_size_x=64)


# =============================================================================
# Factory Methods Tests
# =============================================================================


class TestFromRenderSettings:
    """Tests for factory from RenderSettingsNode."""

    def test_basic_creation(self):
        """Create from RenderSettingsNode."""
        settings = RenderSettingsNode(
            width=800,
            height=600,
            workgroup_size_x=16,
            workgroup_size_y=16,
        )
        dispatch = ComputeDispatch.from_render_settings(settings)
        assert dispatch.width == 800
        assert dispatch.height == 600
        assert dispatch.workgroup_size_x == 16
        assert dispatch.workgroup_size_y == 16

    def test_default_settings(self):
        """Default RenderSettingsNode values."""
        settings = RenderSettingsNode()
        dispatch = ComputeDispatch.from_render_settings(settings)
        assert dispatch.width == 1920
        assert dispatch.height == 1080
        assert dispatch.workgroup_size_x == 8
        assert dispatch.workgroup_size_y == 8


# =============================================================================
# WGSL Generation Tests
# =============================================================================


class TestWGSLGeneration:
    """Tests for WGSL code generation."""

    def test_generate_contains_workgroup_size(self):
        """Generated code should have correct workgroup_size."""
        dispatch = ComputeDispatch(100, 100, 8, 8)
        wgsl = dispatch.generate_dispatch_code()
        assert "@workgroup_size(8, 8, 1)" in wgsl

    def test_generate_contains_bounds_check(self):
        """Generated code should have bounds check."""
        dispatch = ComputeDispatch(100, 100)
        wgsl = dispatch.generate_dispatch_code()
        assert "pixel.x >= VIEWPORT_WIDTH" in wgsl
        assert "pixel.y >= VIEWPORT_HEIGHT" in wgsl
        assert "return;" in wgsl

    def test_generate_contains_uv_calculation(self):
        """Generated code should compute UV."""
        dispatch = ComputeDispatch(100, 100)
        wgsl = dispatch.generate_dispatch_code()
        assert "let uv" in wgsl
        assert "0.5" in wgsl  # Pixel center offset

    def test_generate_contains_viewport_constants(self):
        """Generated code should have viewport constants."""
        dispatch = ComputeDispatch(1920, 1080)
        wgsl = dispatch.generate_dispatch_code()
        assert "VIEWPORT_WIDTH: i32 = 1920" in wgsl
        assert "VIEWPORT_HEIGHT: i32 = 1080" in wgsl

    def test_generate_contains_texture_store(self):
        """Generated code should write to storage texture."""
        dispatch = ComputeDispatch(100, 100)
        wgsl = dispatch.generate_dispatch_code()
        assert "textureStore" in wgsl

    def test_generate_minimal_no_bindings(self):
        """Minimal generation without bindings."""
        dispatch = ComputeDispatch(100, 100)
        wgsl = dispatch.generate_dispatch_code(
            include_bindings=False,
            include_uniforms=False,
        )
        assert "@group(0)" not in wgsl
        assert "struct Uniforms" not in wgsl

    def test_generate_with_sky_color(self):
        """Generation with sky color call."""
        dispatch = ComputeDispatch(100, 100)
        wgsl = dispatch.generate_dispatch_code(include_sky_color=True)
        assert "sky_color" in wgsl

    def test_generate_without_sky_color(self):
        """Generation without sky color."""
        dispatch = ComputeDispatch(100, 100)
        wgsl = dispatch.generate_dispatch_code(include_sky_color=False)
        # Should have default background instead
        assert "vec3<f32>(0.1, 0.1, 0.15)" in wgsl

    def test_generate_custom_workgroup(self):
        """Custom workgroup size in generated code."""
        dispatch = ComputeDispatch(100, 100, 16, 4)
        wgsl = dispatch.generate_dispatch_code()
        assert "@workgroup_size(16, 4, 1)" in wgsl


class TestEntryPointTemplate:
    """Tests for entry point template generation."""

    def test_template_basic(self):
        """Basic template generation."""
        template = generate_entry_point_template(1920, 1080)
        assert "@compute @workgroup_size(8, 8, 1)" in template
        assert "1920" in template
        assert "1080" in template

    def test_template_custom_workgroup(self):
        """Template with custom workgroup."""
        template = generate_entry_point_template(100, 100, 16, 16)
        assert "@workgroup_size(16, 16, 1)" in template


# =============================================================================
# Bind Group Configuration Tests
# =============================================================================


class TestBindGroupConfig:
    """Tests for BindGroupConfig."""

    def test_default_format(self):
        """Default output format."""
        config = BindGroupConfig()
        assert config.output_format == OutputFormat.RGBA8_UNORM

    def test_generate_output_binding(self):
        """Generate output texture binding."""
        config = BindGroupConfig(output_format=OutputFormat.RGBA16_FLOAT)
        binding = config.generate_output_binding()
        assert "rgba16float" in binding
        assert "@group(0) @binding(0)" in binding

    def test_generate_uniforms_struct(self):
        """Generate uniforms struct."""
        config = BindGroupConfig()
        uniforms = config.generate_uniforms_struct()
        assert "struct Uniforms" in uniforms
        assert "camera_origin" in uniforms
        assert "camera_fov" in uniforms
        assert "max_steps" in uniforms


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_rgba8_value(self):
        """RGBA8 format value."""
        assert OutputFormat.RGBA8_UNORM.value == "rgba8unorm"

    def test_rgba16_value(self):
        """RGBA16 format value."""
        assert OutputFormat.RGBA16_FLOAT.value == "rgba16float"

    def test_rgba32_value(self):
        """RGBA32 format value."""
        assert OutputFormat.RGBA32_FLOAT.value == "rgba32float"

    def test_r32_value(self):
        """R32 format value."""
        assert OutputFormat.R32_FLOAT.value == "r32float"
