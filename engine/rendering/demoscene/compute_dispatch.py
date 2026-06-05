"""
Full-Screen Compute Shader Dispatch for SDF Ray Marching (T-DEMO-3.9).

This module implements compute shader dispatch configuration and WGSL code
generation for full-screen ray marching. It handles:
  - Workgroup size configuration (default 8x8x1)
  - Dispatch dimension calculation (ceil division)
  - UV coordinate computation for pixel centers
  - Bounds checking for partial workgroups

The compute dispatch model:
  - Each thread processes one pixel
  - Workgroups are 2D (8x8 by default) for GPU cache efficiency
  - Dispatch dimensions = ceil(resolution / workgroup_size)
  - Edge workgroups skip out-of-bounds threads

Usage:
    >>> from engine.rendering.demoscene.compute_dispatch import ComputeDispatch
    >>> dispatch = ComputeDispatch(1920, 1080)
    >>> wgsl = dispatch.generate_dispatch_code()
    >>> dispatch_x, dispatch_y = dispatch.dispatch_dimensions()
    >>> dispatch_x, dispatch_y
    (240, 135)

Reference:
    - Scene codegen: scene_codegen.py (integrates dispatch into full shader)
    - Ray generation: ray_generation.py (ray_march called per-pixel)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .ast_nodes import RenderSettingsNode, Vec3Node


# =============================================================================
# Output Format Enumeration
# =============================================================================


class OutputFormat(Enum):
    """Storage texture output formats."""
    RGBA8_UNORM = "rgba8unorm"
    RGBA16_FLOAT = "rgba16float"
    RGBA32_FLOAT = "rgba32float"
    R32_FLOAT = "r32float"


# =============================================================================
# Bind Group Layout Configuration
# =============================================================================


@dataclass(frozen=True)
class BindGroupConfig:
    """Configuration for compute shader bind groups.

    Defines the binding layout for:
      - Group 0: Output storage texture
      - Group 1: Uniforms (camera, render settings, time)

    Attributes:
        output_format: Storage texture format.
        output_binding: Binding index for output texture (default 0).
        uniforms_binding: Binding index for uniform buffer (default 0).
    """
    output_format: OutputFormat = OutputFormat.RGBA8_UNORM
    output_binding: int = 0
    uniforms_binding: int = 0

    def generate_output_binding(self) -> str:
        """Generate WGSL for output texture binding."""
        return (
            f"@group(0) @binding({self.output_binding})\n"
            f"var output_texture: texture_storage_2d<{self.output_format.value}, write>;"
        )

    def generate_uniforms_struct(self) -> str:
        """Generate WGSL for uniforms struct and binding."""
        return """\
struct Uniforms {
    camera_origin: vec3<f32>,
    camera_fov: f32,
    camera_target: vec3<f32>,
    camera_aspect: f32,
    camera_up: vec3<f32>,
    time: f32,
    resolution: vec2<f32>,
    max_steps: u32,
    max_distance: f32,
    epsilon: f32,
    _padding: vec3<f32>,
}

@group(1) @binding(0)
var<uniform> uniforms: Uniforms;"""


# =============================================================================
# Compute Dispatch Configuration
# =============================================================================


@dataclass
class ComputeDispatch:
    """
    Configuration and code generation for full-screen compute dispatch.

    This class handles:
      - Workgroup size configuration (default 8x8x1)
      - Dispatch dimension calculation using ceiling division
      - WGSL entry point generation with pixel coordinate mapping
      - Bounds checking for edge workgroups

    The coordinate system:
      - global_invocation_id.xy maps directly to pixel coordinates
      - UV coordinates are computed for pixel centers: (pixel + 0.5) / resolution
      - UV is normalized to [-1, 1] for ray generation

    Attributes:
        width: Viewport width in pixels.
        height: Viewport height in pixels.
        workgroup_size_x: Workgroup X dimension (default 8).
        workgroup_size_y: Workgroup Y dimension (default 8).
        output_format: Storage texture format.
    """
    width: int
    height: int
    workgroup_size_x: int = 8
    workgroup_size_y: int = 8
    output_format: OutputFormat = OutputFormat.RGBA8_UNORM

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Height must be positive, got {self.height}")
        if self.workgroup_size_x <= 0:
            raise ValueError(f"Workgroup X must be positive, got {self.workgroup_size_x}")
        if self.workgroup_size_y <= 0:
            raise ValueError(f"Workgroup Y must be positive, got {self.workgroup_size_y}")
        # Workgroup size should be power of 2 for best performance
        if self.workgroup_size_x > 32 or self.workgroup_size_y > 32:
            raise ValueError(
                f"Workgroup dimensions should not exceed 32, "
                f"got ({self.workgroup_size_x}, {self.workgroup_size_y})"
            )

    @classmethod
    def from_render_settings(cls, settings: RenderSettingsNode) -> "ComputeDispatch":
        """Create ComputeDispatch from RenderSettingsNode.

        Args:
            settings: Render settings containing resolution and workgroup size.

        Returns:
            Configured ComputeDispatch instance.
        """
        return cls(
            width=settings.width,
            height=settings.height,
            workgroup_size_x=settings.workgroup_size_x,
            workgroup_size_y=settings.workgroup_size_y,
        )

    def dispatch_dimensions(self) -> Tuple[int, int]:
        """
        Calculate dispatch dimensions using ceiling division.

        The dispatch covers the entire viewport with enough workgroups
        such that every pixel has a corresponding thread.

        Returns:
            Tuple of (dispatch_x, dispatch_y) workgroup counts.

        Example:
            >>> dispatch = ComputeDispatch(1920, 1080)
            >>> dispatch.dispatch_dimensions()
            (240, 135)
            >>> # 240 * 8 = 1920, 135 * 8 = 1080 (exact fit)

            >>> dispatch = ComputeDispatch(1000, 600, 8, 8)
            >>> dispatch.dispatch_dimensions()
            (125, 75)
            >>> # 125 * 8 = 1000, 75 * 8 = 600 (exact fit)

            >>> dispatch = ComputeDispatch(1001, 601, 8, 8)
            >>> dispatch.dispatch_dimensions()
            (126, 76)
            >>> # 126 * 8 = 1008 > 1001, 76 * 8 = 608 > 601 (partial workgroups)
        """
        dispatch_x = (self.width + self.workgroup_size_x - 1) // self.workgroup_size_x
        dispatch_y = (self.height + self.workgroup_size_y - 1) // self.workgroup_size_y
        return (dispatch_x, dispatch_y)

    def total_threads(self) -> int:
        """Return total number of threads that will be dispatched.

        Note: This may be larger than width * height due to partial workgroups.
        """
        dispatch_x, dispatch_y = self.dispatch_dimensions()
        return dispatch_x * self.workgroup_size_x * dispatch_y * self.workgroup_size_y

    def pixel_to_uv(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to normalized UV coordinates.

        UV coordinates are computed for pixel centers (adding 0.5)
        and normalized to [-1, 1] range for ray generation.

        Args:
            x: Pixel X coordinate (0 to width-1).
            y: Pixel Y coordinate (0 to height-1).

        Returns:
            Tuple (u, v) in [-1, 1] range.

        Example:
            >>> dispatch = ComputeDispatch(100, 100)
            >>> dispatch.pixel_to_uv(0, 0)  # Top-left
            (-0.99, 0.99)
            >>> dispatch.pixel_to_uv(99, 99)  # Bottom-right
            (0.99, -0.99)
            >>> dispatch.pixel_to_uv(49, 49)  # Near center
            (-0.01, 0.01)
        """
        # Add 0.5 to sample pixel centers
        u = ((x + 0.5) / self.width) * 2.0 - 1.0
        v = 1.0 - ((y + 0.5) / self.height) * 2.0  # Flip Y
        return (u, v)

    def uv_to_pixel(self, u: float, v: float) -> Tuple[int, int]:
        """
        Convert UV coordinates back to pixel coordinates.

        Args:
            u: Horizontal coordinate in [-1, 1].
            v: Vertical coordinate in [-1, 1].

        Returns:
            Tuple (x, y) pixel coordinates.
        """
        x = int(((u + 1.0) / 2.0) * self.width)
        y = int(((1.0 - v) / 2.0) * self.height)
        # Clamp to valid range
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        return (x, y)

    def is_in_bounds(self, x: int, y: int) -> bool:
        """Check if pixel coordinates are within the viewport."""
        return 0 <= x < self.width and 0 <= y < self.height

    def generate_dispatch_code(
        self,
        *,
        include_bindings: bool = True,
        include_uniforms: bool = True,
        include_ray_march_call: bool = True,
        include_sky_color: bool = True,
    ) -> str:
        """
        Generate WGSL compute shader entry point.

        Generates the @compute entry point function that:
          1. Computes pixel coordinates from global_invocation_id
          2. Skips out-of-bounds pixels (partial workgroups)
          3. Converts pixel to UV coordinates
          4. Generates ray and calls ray_march
          5. Shades hit points or uses sky color for misses
          6. Writes result to storage texture

        Args:
            include_bindings: Include bind group declarations.
            include_uniforms: Include uniform buffer struct.
            include_ray_march_call: Include ray march and shading logic.
            include_sky_color: Include sky color for miss rays.

        Returns:
            WGSL code string for compute entry point.
        """
        lines: list[str] = []

        # Header
        lines.append("// Auto-generated compute dispatch (T-DEMO-3.9)")
        lines.append(f"// Resolution: {self.width}x{self.height}")
        lines.append(f"// Workgroup: {self.workgroup_size_x}x{self.workgroup_size_y}x1")
        dispatch_x, dispatch_y = self.dispatch_dimensions()
        lines.append(f"// Dispatch: {dispatch_x}x{dispatch_y}")
        lines.append("")

        # Bindings
        if include_bindings:
            config = BindGroupConfig(output_format=self.output_format)
            lines.append(config.generate_output_binding())
            lines.append("")

        # Uniforms
        if include_uniforms:
            config = BindGroupConfig()
            lines.append(config.generate_uniforms_struct())
            lines.append("")

        # Constants
        lines.append(f"const VIEWPORT_WIDTH: i32 = {self.width};")
        lines.append(f"const VIEWPORT_HEIGHT: i32 = {self.height};")
        lines.append(f"const VIEWPORT_WIDTH_F: f32 = {float(self.width)};")
        lines.append(f"const VIEWPORT_HEIGHT_F: f32 = {float(self.height)};")
        lines.append("")

        # Main entry point
        lines.append("/// Main compute shader entry point.")
        lines.append("/// Each thread processes one pixel of the output image.")
        lines.append(
            f"@compute @workgroup_size({self.workgroup_size_x}, {self.workgroup_size_y}, 1)"
        )
        lines.append("fn main(@builtin(global_invocation_id) gid: vec3<u32>) {")
        lines.append("    // Convert to signed pixel coordinates")
        lines.append("    let pixel = vec2<i32>(gid.xy);")
        lines.append("")
        lines.append("    // Bounds check for partial workgroups")
        lines.append("    if (pixel.x >= VIEWPORT_WIDTH || pixel.y >= VIEWPORT_HEIGHT) {")
        lines.append("        return;")
        lines.append("    }")
        lines.append("")
        lines.append("    // Compute UV coordinates for pixel center")
        lines.append("    // UV is in [-1, 1] with Y flipped (top = +1)")
        lines.append("    let uv = vec2<f32>(")
        lines.append("        (f32(pixel.x) + 0.5) / VIEWPORT_WIDTH_F * 2.0 - 1.0,")
        lines.append("        1.0 - (f32(pixel.y) + 0.5) / VIEWPORT_HEIGHT_F * 2.0")
        lines.append("    );")
        lines.append("")

        if include_ray_march_call:
            lines.append("    // Generate camera ray")
            lines.append("    let ray = generate_ray(")
            lines.append("        uv,")
            lines.append("        uniforms.camera_origin,")
            lines.append("        uniforms.camera_target,")
            lines.append("        uniforms.camera_up,")
            lines.append("        uniforms.camera_fov,")
            lines.append("        uniforms.camera_aspect,")
            lines.append("    );")
            lines.append("")
            lines.append("    // Ray march through scene")
            lines.append("    let hit = ray_march(")
            lines.append("        uniforms.camera_origin,")
            lines.append("        ray,")
            lines.append("        uniforms.max_steps,")
            lines.append("        uniforms.max_distance,")
            lines.append("        uniforms.epsilon,")
            lines.append("    );")
            lines.append("")
            lines.append("    var color: vec3<f32>;")
            lines.append("    if (hit.hit) {")
            lines.append("        // Shade hit point")
            lines.append("        let normal = estimate_normal(hit.position, uniforms.epsilon);")
            lines.append("        let material = scene_material(hit.material_id);")
            lines.append("        let view_dir = normalize(uniforms.camera_origin - hit.position);")
            lines.append("        color = calculate_lighting(hit.position, normal, view_dir, material);")
            lines.append("        color = tone_map_aces(color);")
            lines.append("        color = linear_to_srgb(color);")
            lines.append("    } else {")

            if include_sky_color:
                lines.append("        // Sky color for miss")
                lines.append("        color = sky_color(ray);")
            else:
                lines.append("        // Default background")
                lines.append("        color = vec3<f32>(0.1, 0.1, 0.15);")

            lines.append("    }")
            lines.append("")
            lines.append("    // Write to output texture")
            lines.append("    textureStore(output_texture, pixel, vec4<f32>(color, 1.0));")
        else:
            lines.append("    // Placeholder: write UV as color for debugging")
            lines.append("    let debug_color = vec3<f32>((uv.x + 1.0) * 0.5, (uv.y + 1.0) * 0.5, 0.5);")
            lines.append("    textureStore(output_texture, pixel, vec4<f32>(debug_color, 1.0));")

        lines.append("}")

        return "\n".join(lines)

    def generate_dispatch_code_minimal(self) -> str:
        """
        Generate minimal WGSL entry point (no dependencies).

        Returns only the entry point function assuming all helper functions
        (generate_ray, ray_march, etc.) are defined elsewhere.
        """
        return self.generate_dispatch_code(
            include_bindings=False,
            include_uniforms=False,
            include_ray_march_call=True,
            include_sky_color=True,
        )


# =============================================================================
# Entry Point Template Functions
# =============================================================================


def generate_entry_point_template(
    width: int,
    height: int,
    workgroup_x: int = 8,
    workgroup_y: int = 8,
) -> str:
    """
    Generate WGSL entry point template with placeholders.

    This template uses `WIDTH` and `HEIGHT` constants that should be
    defined or replaced in the final shader.

    Args:
        width: Viewport width.
        height: Viewport height.
        workgroup_x: Workgroup X size.
        workgroup_y: Workgroup Y size.

    Returns:
        WGSL entry point template.
    """
    return f"""\
@compute @workgroup_size({workgroup_x}, {workgroup_y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let pixel = vec2<i32>(gid.xy);
    if (pixel.x >= {width} || pixel.y >= {height}) {{ return; }}

    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>({float(width)}, {float(height)});
    let ray = generate_ray(uv);
    let hit = ray_march(ray);

    var color: vec3<f32>;
    if (hit.hit) {{
        color = shade_point(hit);
    }} else {{
        color = sky_color(ray.direction);
    }}

    textureStore(output_texture, pixel, vec4<f32>(color, 1.0));
}}
"""


def calculate_dispatch_dimensions(
    width: int,
    height: int,
    workgroup_x: int = 8,
    workgroup_y: int = 8,
) -> Tuple[int, int]:
    """
    Calculate dispatch dimensions for given resolution.

    This is a convenience function that computes ceiling division
    without creating a ComputeDispatch instance.

    Args:
        width: Viewport width.
        height: Viewport height.
        workgroup_x: Workgroup X size.
        workgroup_y: Workgroup Y size.

    Returns:
        Tuple of (dispatch_x, dispatch_y).
    """
    dispatch_x = (width + workgroup_x - 1) // workgroup_x
    dispatch_y = (height + workgroup_y - 1) // workgroup_y
    return (dispatch_x, dispatch_y)


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "ComputeDispatch",
    "BindGroupConfig",
    "OutputFormat",
    "generate_entry_point_template",
    "calculate_dispatch_dimensions",
]
