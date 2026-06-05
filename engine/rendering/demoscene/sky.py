"""
Sky Color Functions for SDF Ray Marching (T-DEMO-3.10).

This module implements sky color calculation for rays that miss scene geometry.
It provides both Python evaluation for testing and WGSL code generation
for GPU compute shader integration.

Sky color modes:
  - Gradient: Smooth blend from horizon to zenith based on ray Y direction
  - Solid: Constant color background
  - Procedural: Time-based animated sky (for demoscene effects)

The sky color function is called when ray_march returns hit=false.
The ray direction's Y component determines the vertical position:
  - Y = -1: Looking down (horizon or below-horizon color)
  - Y = 0: Horizon
  - Y = +1: Looking up (zenith color)

Usage:
    >>> from engine.rendering.demoscene.sky import SkyConfig, SkyMode
    >>> sky = SkyConfig(
    ...     mode=SkyMode.GRADIENT,
    ...     horizon_color=(0.8, 0.6, 0.3),
    ...     zenith_color=(0.1, 0.2, 0.5),
    ... )
    >>> wgsl = sky.generate_sky_wgsl()
    >>> "sky_color" in wgsl
    True

Reference:
    - compute_dispatch.py: Calls sky_color() for miss rays
    - scene_codegen.py: Integrates sky into full shader
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Union

from .ast_nodes import Vec3Node, FloatNode


# =============================================================================
# Sky Mode Enumeration
# =============================================================================


class SkyMode(Enum):
    """Sky rendering modes."""
    SOLID = "solid"
    GRADIENT = "gradient"
    GRADIENT_TRIPLE = "gradient_triple"
    PROCEDURAL = "procedural"


# =============================================================================
# Vec3 Helper (for Python evaluation)
# =============================================================================


@dataclass
class Vec3:
    """Simple 3D vector for sky color calculations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "Vec3":
        return cls(t[0], t[1], t[2])

    @classmethod
    def from_node(cls, node: Vec3Node) -> "Vec3":
        return cls(float(node.x), float(node.y), float(node.z))

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def lerp(self, other: "Vec3", t: float) -> "Vec3":
        """Linear interpolation between self and other."""
        return self * (1.0 - t) + other * t

    def clamp(self, min_val: float = 0.0, max_val: float = 1.0) -> "Vec3":
        """Clamp all components to range."""
        return Vec3(
            max(min_val, min(max_val, self.x)),
            max(min_val, min(max_val, self.y)),
            max(min_val, min(max_val, self.z)),
        )


# =============================================================================
# Sky Color Functions (Python Evaluation)
# =============================================================================


def sky_solid(direction: Vec3, color: Vec3) -> Vec3:
    """
    Solid color sky (ignores direction).

    Args:
        direction: Ray direction (unused).
        color: Solid sky color.

    Returns:
        The solid color.
    """
    return color


def sky_gradient(
    direction: Vec3,
    horizon_color: Vec3,
    zenith_color: Vec3,
) -> Vec3:
    """
    Two-color gradient sky based on ray Y direction.

    The blend factor is computed from the ray direction's Y component:
      - Y = -1 to 0: Below horizon, uses horizon color
      - Y = 0 to +1: Interpolates from horizon to zenith

    Args:
        direction: Normalized ray direction.
        horizon_color: Color at the horizon (Y=0).
        zenith_color: Color at the zenith (Y=+1).

    Returns:
        Interpolated sky color.
    """
    # Map Y from [-1, 1] to [0, 1], clamping below horizon
    t = max(0.0, direction.y)
    return horizon_color.lerp(zenith_color, t)


def sky_gradient_triple(
    direction: Vec3,
    below_horizon_color: Vec3,
    horizon_color: Vec3,
    zenith_color: Vec3,
) -> Vec3:
    """
    Three-color gradient sky with separate below-horizon color.

    Args:
        direction: Normalized ray direction.
        below_horizon_color: Color for Y < 0.
        horizon_color: Color at Y = 0.
        zenith_color: Color at Y = +1.

    Returns:
        Interpolated sky color.
    """
    if direction.y < 0.0:
        # Below horizon: blend from below-horizon to horizon
        t = direction.y + 1.0  # Map [-1, 0] to [0, 1]
        return below_horizon_color.lerp(horizon_color, t)
    else:
        # Above horizon: blend from horizon to zenith
        return horizon_color.lerp(zenith_color, direction.y)


def sky_procedural(
    direction: Vec3,
    time: float,
    base_color: Vec3,
    sun_color: Vec3,
    sun_direction: Vec3,
) -> Vec3:
    """
    Procedural sky with sun disc and time-based variation.

    Args:
        direction: Normalized ray direction.
        time: Time value for animation.
        base_color: Base sky gradient color.
        sun_color: Sun disc color.
        sun_direction: Normalized sun direction.

    Returns:
        Sky color with sun contribution.
    """
    # Base gradient
    t = max(0.0, direction.y)
    sky = base_color.lerp(Vec3(0.1, 0.2, 0.4), t)

    # Sun disc (simple dot product falloff)
    sun_dot = (
        direction.x * sun_direction.x +
        direction.y * sun_direction.y +
        direction.z * sun_direction.z
    )
    sun_intensity = max(0.0, sun_dot) ** 64.0

    # Add sun contribution
    result = sky + sun_color * sun_intensity
    return result.clamp()


# =============================================================================
# Sky Configuration Dataclass
# =============================================================================


@dataclass
class SkyConfig:
    """
    Configuration for sky color rendering.

    This class defines all parameters needed for sky color calculation
    and generates corresponding WGSL code.

    Attributes:
        mode: Sky rendering mode (solid, gradient, etc.).
        solid_color: Color for solid mode.
        horizon_color: Color at horizon for gradient modes.
        zenith_color: Color at zenith for gradient modes.
        below_horizon_color: Color below horizon for triple gradient.
        sun_enabled: Enable procedural sun disc.
        sun_color: Sun disc color.
        sun_direction: Normalized sun direction vector.
        sun_power: Sun falloff exponent (higher = smaller sun).
    """
    mode: SkyMode = SkyMode.GRADIENT
    solid_color: Tuple[float, float, float] = (0.1, 0.1, 0.15)
    horizon_color: Tuple[float, float, float] = (0.8, 0.6, 0.4)
    zenith_color: Tuple[float, float, float] = (0.1, 0.2, 0.5)
    below_horizon_color: Tuple[float, float, float] = (0.05, 0.05, 0.08)
    sun_enabled: bool = False
    sun_color: Tuple[float, float, float] = (1.0, 0.95, 0.8)
    sun_direction: Tuple[float, float, float] = (0.5, 0.7, 0.5)
    sun_power: float = 64.0

    def evaluate(self, direction: Vec3, time: float = 0.0) -> Vec3:
        """
        Evaluate sky color for a given direction.

        Args:
            direction: Normalized ray direction.
            time: Time for procedural effects.

        Returns:
            Sky color as Vec3.
        """
        if self.mode == SkyMode.SOLID:
            return sky_solid(direction, Vec3.from_tuple(self.solid_color))

        elif self.mode == SkyMode.GRADIENT:
            color = sky_gradient(
                direction,
                Vec3.from_tuple(self.horizon_color),
                Vec3.from_tuple(self.zenith_color),
            )

        elif self.mode == SkyMode.GRADIENT_TRIPLE:
            color = sky_gradient_triple(
                direction,
                Vec3.from_tuple(self.below_horizon_color),
                Vec3.from_tuple(self.horizon_color),
                Vec3.from_tuple(self.zenith_color),
            )

        elif self.mode == SkyMode.PROCEDURAL:
            color = sky_procedural(
                direction,
                time,
                Vec3.from_tuple(self.horizon_color),
                Vec3.from_tuple(self.sun_color),
                Vec3.from_tuple(self.sun_direction),
            )

        else:
            color = Vec3.from_tuple(self.solid_color)

        # Add sun if enabled and not already in procedural mode
        if self.sun_enabled and self.mode != SkyMode.PROCEDURAL:
            color = self._add_sun(direction, color)

        return color

    def _add_sun(self, direction: Vec3, base_color: Vec3) -> Vec3:
        """Add sun disc to base sky color."""
        sun_dir = Vec3.from_tuple(self.sun_direction)
        sun_col = Vec3.from_tuple(self.sun_color)

        # Normalize sun direction
        length = math.sqrt(sun_dir.x**2 + sun_dir.y**2 + sun_dir.z**2)
        if length > 1e-6:
            sun_dir = sun_dir * (1.0 / length)

        # Sun disc
        dot = (
            direction.x * sun_dir.x +
            direction.y * sun_dir.y +
            direction.z * sun_dir.z
        )
        intensity = max(0.0, dot) ** self.sun_power

        result = base_color + sun_col * intensity
        return result.clamp()

    def generate_sky_wgsl(
        self,
        *,
        function_name: str = "sky_color",
        include_sun: bool = True,
    ) -> str:
        """
        Generate WGSL code for sky color function.

        Args:
            function_name: Name of the generated function.
            include_sun: Include sun disc calculation if enabled.

        Returns:
            WGSL code string.
        """
        lines: list[str] = []

        # Header comment
        lines.append(f"/// Sky color function ({self.mode.value} mode).")
        lines.append(f"/// Generated by SkyConfig (T-DEMO-3.10).")

        if self.mode == SkyMode.SOLID:
            lines.append(self._generate_solid_wgsl(function_name))

        elif self.mode == SkyMode.GRADIENT:
            lines.append(self._generate_gradient_wgsl(function_name, include_sun))

        elif self.mode == SkyMode.GRADIENT_TRIPLE:
            lines.append(self._generate_triple_gradient_wgsl(function_name, include_sun))

        elif self.mode == SkyMode.PROCEDURAL:
            lines.append(self._generate_procedural_wgsl(function_name))

        return "\n".join(lines)

    def _generate_solid_wgsl(self, fn_name: str) -> str:
        """Generate solid color sky WGSL."""
        c = self.solid_color
        return f"""\
fn {fn_name}(direction: vec3<f32>) -> vec3<f32> {{
    return vec3<f32>({c[0]}, {c[1]}, {c[2]});
}}
"""

    def _generate_gradient_wgsl(self, fn_name: str, include_sun: bool) -> str:
        """Generate two-color gradient sky WGSL."""
        h = self.horizon_color
        z = self.zenith_color

        code = f"""\
fn {fn_name}(direction: vec3<f32>) -> vec3<f32> {{
    // Blend factor from horizon (Y=0) to zenith (Y=1)
    let t = max(0.0, direction.y);
    let horizon = vec3<f32>({h[0]}, {h[1]}, {h[2]});
    let zenith = vec3<f32>({z[0]}, {z[1]}, {z[2]});
    var color = mix(horizon, zenith, t);
"""

        if include_sun and self.sun_enabled:
            code += self._generate_sun_wgsl_inline()

        code += """\
    return color;
}
"""
        return code

    def _generate_triple_gradient_wgsl(self, fn_name: str, include_sun: bool) -> str:
        """Generate three-color gradient sky WGSL."""
        b = self.below_horizon_color
        h = self.horizon_color
        z = self.zenith_color

        code = f"""\
fn {fn_name}(direction: vec3<f32>) -> vec3<f32> {{
    let below = vec3<f32>({b[0]}, {b[1]}, {b[2]});
    let horizon = vec3<f32>({h[0]}, {h[1]}, {h[2]});
    let zenith = vec3<f32>({z[0]}, {z[1]}, {z[2]});

    var color: vec3<f32>;
    if (direction.y < 0.0) {{
        // Below horizon
        let t = direction.y + 1.0;  // Map [-1, 0] to [0, 1]
        color = mix(below, horizon, t);
    }} else {{
        // Above horizon
        color = mix(horizon, zenith, direction.y);
    }}
"""

        if include_sun and self.sun_enabled:
            code += self._generate_sun_wgsl_inline()

        code += """\
    return color;
}
"""
        return code

    def _generate_procedural_wgsl(self, fn_name: str) -> str:
        """Generate procedural sky WGSL with sun."""
        h = self.horizon_color
        s = self.sun_color
        sd = self.sun_direction

        return f"""\
fn {fn_name}(direction: vec3<f32>) -> vec3<f32> {{
    // Time-based sky gradient
    let t = max(0.0, direction.y);
    let base = vec3<f32>({h[0]}, {h[1]}, {h[2]});
    let top = vec3<f32>(0.1, 0.2, 0.4);
    var color = mix(base, top, t);

    // Sun disc
    let sun_dir = normalize(vec3<f32>({sd[0]}, {sd[1]}, {sd[2]}));
    let sun_color = vec3<f32>({s[0]}, {s[1]}, {s[2]});
    let sun_dot = max(0.0, dot(direction, sun_dir));
    let sun_intensity = pow(sun_dot, {self.sun_power});

    color = color + sun_color * sun_intensity;
    return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}}
"""

    def _generate_sun_wgsl_inline(self) -> str:
        """Generate inline sun calculation WGSL."""
        s = self.sun_color
        sd = self.sun_direction

        return f"""
    // Sun disc
    let sun_dir = normalize(vec3<f32>({sd[0]}, {sd[1]}, {sd[2]}));
    let sun_color = vec3<f32>({s[0]}, {s[1]}, {s[2]});
    let sun_dot = max(0.0, dot(direction, sun_dir));
    let sun_intensity = pow(sun_dot, {self.sun_power});
    color = clamp(color + sun_color * sun_intensity, vec3<f32>(0.0), vec3<f32>(1.0));
"""


# =============================================================================
# RenderSettingsNode Extension
# =============================================================================


@dataclass
class SkySettingsNode:
    """
    Sky settings for integration with RenderSettingsNode.

    Attributes:
        mode: Sky mode.
        horizon_color: Horizon color node.
        zenith_color: Zenith color node.
        solid_color: Solid color node (for SOLID mode).
        sun_direction: Sun direction node.
        sun_color: Sun color node.
        sun_power: Sun power (falloff exponent).
        sun_enabled: Whether sun is enabled.
    """
    mode: SkyMode = SkyMode.GRADIENT
    horizon_color: Vec3Node = field(
        default_factory=lambda: Vec3Node(0.8, 0.6, 0.4)
    )
    zenith_color: Vec3Node = field(
        default_factory=lambda: Vec3Node(0.1, 0.2, 0.5)
    )
    solid_color: Vec3Node = field(
        default_factory=lambda: Vec3Node(0.1, 0.1, 0.15)
    )
    sun_direction: Vec3Node = field(
        default_factory=lambda: Vec3Node(0.5, 0.7, 0.5)
    )
    sun_color: Vec3Node = field(
        default_factory=lambda: Vec3Node(1.0, 0.95, 0.8)
    )
    sun_power: FloatNode = field(default_factory=lambda: FloatNode(64.0))
    sun_enabled: bool = False

    def to_sky_config(self) -> SkyConfig:
        """Convert to SkyConfig for code generation."""
        return SkyConfig(
            mode=self.mode,
            solid_color=self.solid_color.as_tuple(),
            horizon_color=self.horizon_color.as_tuple(),
            zenith_color=self.zenith_color.as_tuple(),
            sun_enabled=self.sun_enabled,
            sun_color=self.sun_color.as_tuple(),
            sun_direction=self.sun_direction.as_tuple(),
            sun_power=float(self.sun_power.value),
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_sky_wgsl(
    mode: SkyMode = SkyMode.GRADIENT,
    horizon_color: Tuple[float, float, float] = (0.8, 0.6, 0.4),
    zenith_color: Tuple[float, float, float] = (0.1, 0.2, 0.5),
    solid_color: Tuple[float, float, float] = (0.1, 0.1, 0.15),
    function_name: str = "sky_color",
) -> str:
    """
    Generate WGSL sky color function.

    Args:
        mode: Sky mode (solid, gradient, etc.).
        horizon_color: Horizon color RGB tuple.
        zenith_color: Zenith color RGB tuple.
        solid_color: Solid color RGB tuple (for SOLID mode).
        function_name: Name of generated function.

    Returns:
        WGSL code string.
    """
    config = SkyConfig(
        mode=mode,
        horizon_color=horizon_color,
        zenith_color=zenith_color,
        solid_color=solid_color,
    )
    return config.generate_sky_wgsl(function_name=function_name)


def create_sunset_sky() -> SkyConfig:
    """Create a preset sunset sky configuration."""
    return SkyConfig(
        mode=SkyMode.GRADIENT_TRIPLE,
        below_horizon_color=(0.1, 0.05, 0.1),
        horizon_color=(0.9, 0.4, 0.2),
        zenith_color=(0.1, 0.1, 0.3),
        sun_enabled=True,
        sun_color=(1.0, 0.6, 0.1),
        sun_direction=(0.3, 0.2, 0.9),
        sun_power=32.0,
    )


def create_daytime_sky() -> SkyConfig:
    """Create a preset daytime sky configuration."""
    return SkyConfig(
        mode=SkyMode.GRADIENT,
        horizon_color=(0.6, 0.7, 0.9),
        zenith_color=(0.2, 0.4, 0.8),
        sun_enabled=True,
        sun_color=(1.0, 0.98, 0.9),
        sun_direction=(0.5, 0.8, 0.3),
        sun_power=128.0,
    )


def create_night_sky() -> SkyConfig:
    """Create a preset night sky configuration."""
    return SkyConfig(
        mode=SkyMode.GRADIENT,
        horizon_color=(0.02, 0.02, 0.05),
        zenith_color=(0.0, 0.0, 0.02),
        sun_enabled=False,
    )


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Enums
    "SkyMode",
    # Classes
    "Vec3",
    "SkyConfig",
    "SkySettingsNode",
    # Functions
    "sky_solid",
    "sky_gradient",
    "sky_gradient_triple",
    "sky_procedural",
    "generate_sky_wgsl",
    # Presets
    "create_sunset_sky",
    "create_daytime_sky",
    "create_night_sky",
]
