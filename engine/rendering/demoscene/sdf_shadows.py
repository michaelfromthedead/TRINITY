"""
SDF Soft Shadows (T-DEMO-3.6) - Quilez's Improved Penumbra Method.

This module implements soft shadow calculation for signed distance field
scenes using Inigo Quilez's improved penumbra technique. Shadows are computed
by marching toward the light and tracking the minimum ratio of SDF distance
to ray distance.

The technique provides:
  - Contact hardening (shadows sharp near contact, soft farther away)
  - Configurable penumbra width via k parameter
  - No light leaking for fully occluded points
  - Efficient single-pass computation

Reference: Inigo Quilez -- Soft shadows in raymarching
    https://iquilezles.org/articles/rmshadows/

Usage:
    >>> from engine.rendering.demoscene.sdf_shadows import (
    ...     calculate_soft_shadow, generate_shadow_wgsl, ShadowConfig
    ... )
    >>> shadow = calculate_soft_shadow(point, light_dir, scene_sdf)
    >>> wgsl = generate_shadow_wgsl(ShadowConfig(k=32.0))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .sdf_ast import SDFNode, Vec3


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class ShadowConfig:
    """Configuration for soft shadow calculation.

    Attributes:
        k: Penumbra sharpness parameter (default 16.0).
           Higher values = harder/sharper shadows.
           Lower values = softer shadows.
           Typical range: 4.0 (very soft) to 64.0 (nearly hard).
        max_steps: Maximum ray marching steps (default 32).
        min_dist: Minimum ray distance (default 0.01).
                  Prevents self-shadowing artifacts.
        max_dist: Maximum ray distance (default 100.0).
        min_step: Minimum step size for clamping (default 0.001).
        max_step: Maximum step size for clamping (default 1.0).
        epsilon: Hit threshold (default 0.0001).
    """
    k: float = 16.0
    max_steps: int = 32
    min_dist: float = 0.01
    max_dist: float = 100.0
    min_step: float = 0.001
    max_step: float = 1.0
    epsilon: float = 0.0001

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.k <= 0.0:
            raise ValueError(f"k must be > 0, got {self.k}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.min_dist < 0.0:
            raise ValueError(f"min_dist must be >= 0, got {self.min_dist}")
        if self.max_dist <= self.min_dist:
            raise ValueError(
                f"max_dist must be > min_dist, got {self.max_dist} <= {self.min_dist}"
            )
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")


# =============================================================================
# Vec3 Helper (for standalone use)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Vec3Local:
    """Local Vec3 for standalone calculations without importing sdf_ast."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3Local") -> "Vec3Local":
        return Vec3Local(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3Local") -> "Vec3Local":
        return Vec3Local(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3Local":
        return Vec3Local(self.x * scalar, self.y * scalar, self.z * scalar)

    def __neg__(self) -> "Vec3Local":
        return Vec3Local(-self.x, -self.y, -self.z)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3Local":
        ln = self.length()
        if ln < 1e-10:
            return Vec3Local(0.0, 0.0, 0.0)
        return Vec3Local(self.x / ln, self.y / ln, self.z / ln)

    def dot(self, other: "Vec3Local") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


# =============================================================================
# Core Soft Shadow Calculation (Python Reference Implementation)
# =============================================================================

def calculate_soft_shadow(
    ro: tuple[float, float, float],
    rd: tuple[float, float, float],
    scene_sdf: Callable[[tuple[float, float, float]], float],
    config: Optional[ShadowConfig] = None,
) -> float:
    """
    Calculate soft shadow factor using Quilez's improved penumbra method.

    The algorithm marches from the surface point toward the light,
    tracking the minimum ratio k*h/t where h is the SDF distance
    and t is the ray distance. This produces contact hardening:
    shadows are sharp near occluders and soft farther away.

    Algorithm (Quilez improved soft shadows):
        res = 1.0
        t = min_dist
        for step in 0..max_steps:
            h = scene_sdf(ro + rd * t)
            if h < epsilon: return 0.0  # in shadow
            res = min(res, k * h / t)
            t += clamp(h, min_step, max_step)
            if t > max_dist: break
        return clamp(res, 0.0, 1.0)

    Args:
        ro: Ray origin (surface point, slightly offset along normal).
        rd: Ray direction (towards light, normalized).
        scene_sdf: Scene SDF function mapping position to signed distance.
        config: Shadow configuration parameters (uses defaults if None).

    Returns:
        Shadow factor in [0.0, 1.0] where:
          - 0.0 = fully in shadow
          - 1.0 = fully lit

    Example:
        >>> def sphere_sdf(p):
        ...     # Sphere at (0, 2, 0) with radius 1
        ...     cx, cy, cz = 0, 2, 0
        ...     dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
        ...     return math.sqrt(dx*dx + dy*dy + dz*dz) - 1.0
        >>> # Point on ground, light from above (blocked by sphere)
        >>> shadow = calculate_soft_shadow(
        ...     (0.0, 0.0, 0.0),
        ...     (0.0, 1.0, 0.0),
        ...     sphere_sdf
        ... )
        >>> assert shadow < 0.5  # Significantly shadowed
    """
    if config is None:
        config = ShadowConfig()

    # Normalize direction
    rd_vec = Vec3Local(rd[0], rd[1], rd[2]).normalized()
    ro_vec = Vec3Local(ro[0], ro[1], ro[2])

    res = 1.0
    t = config.min_dist

    for _ in range(config.max_steps):
        # Current position along ray
        pos = ro_vec + rd_vec * t
        h = scene_sdf(pos.as_tuple())

        # Hard shadow: we hit geometry
        if h < config.epsilon:
            return 0.0

        # Soft shadow: track minimum penumbra ratio
        # Contact hardening: k * h / t
        # When close (small t), small h gives hard shadow
        # When far (large t), same h gives softer shadow
        res = min(res, config.k * h / t)

        # Step forward with clamping
        step = max(config.min_step, min(h, config.max_step))
        t += step

        # Beyond max distance: fully lit
        if t > config.max_dist:
            break

    return max(0.0, min(1.0, res))


def calculate_soft_shadow_improved(
    ro: tuple[float, float, float],
    rd: tuple[float, float, float],
    scene_sdf: Callable[[tuple[float, float, float]], float],
    config: Optional[ShadowConfig] = None,
) -> float:
    """
    Calculate soft shadow using Quilez's improved method (2020 version).

    This version improves the penumbra estimation by using a polynomial
    approximation that provides better gradients and fewer artifacts.

    The improvement uses:
        y = h*h / (2.0 * ph)
        d = sqrt(h*h - y*y)
        res = min(res, k * d / max(0, t - y))

    Where ph is the previous height. This provides more accurate
    penumbra estimation, especially for thin objects.

    Args:
        ro: Ray origin.
        rd: Ray direction (normalized).
        scene_sdf: Scene SDF function.
        config: Shadow configuration.

    Returns:
        Shadow factor [0.0, 1.0].
    """
    if config is None:
        config = ShadowConfig()

    rd_vec = Vec3Local(rd[0], rd[1], rd[2]).normalized()
    ro_vec = Vec3Local(ro[0], ro[1], ro[2])

    res = 1.0
    t = config.min_dist
    ph = 1e10  # Previous height (large initial value)

    for _ in range(config.max_steps):
        pos = ro_vec + rd_vec * t
        h = scene_sdf(pos.as_tuple())

        # Hard shadow
        if h < config.epsilon:
            return 0.0

        # Improved penumbra estimation
        y = h * h / (2.0 * ph)
        d = math.sqrt(max(0.0, h * h - y * y))
        denom = max(0.0001, t - y)
        res = min(res, config.k * d / denom)

        # Update previous height
        ph = h

        # Step forward
        step = max(config.min_step, min(h, config.max_step))
        t += step

        if t > config.max_dist:
            break

    return max(0.0, min(1.0, res))


def calculate_hard_shadow(
    ro: tuple[float, float, float],
    rd: tuple[float, float, float],
    scene_sdf: Callable[[tuple[float, float, float]], float],
    config: Optional[ShadowConfig] = None,
) -> float:
    """
    Calculate hard shadow (binary: 0 or 1).

    Simpler than soft shadows, just checks if the ray hits geometry.

    Args:
        ro: Ray origin.
        rd: Ray direction.
        scene_sdf: Scene SDF function.
        config: Shadow configuration (only uses max_steps, distances, epsilon).

    Returns:
        0.0 if in shadow, 1.0 if lit.
    """
    if config is None:
        config = ShadowConfig()

    rd_vec = Vec3Local(rd[0], rd[1], rd[2]).normalized()
    ro_vec = Vec3Local(ro[0], ro[1], ro[2])

    t = config.min_dist

    for _ in range(config.max_steps):
        pos = ro_vec + rd_vec * t
        h = scene_sdf(pos.as_tuple())

        if h < config.epsilon:
            return 0.0

        t += max(config.min_step, h)

        if t > config.max_dist:
            break

    return 1.0


# =============================================================================
# Light-to-Point Shadow Calculation
# =============================================================================

def calculate_shadow_from_light(
    p: tuple[float, float, float],
    n: tuple[float, float, float],
    light_pos: tuple[float, float, float],
    scene_sdf: Callable[[tuple[float, float, float]], float],
    config: Optional[ShadowConfig] = None,
    normal_offset: float = 0.01,
) -> float:
    """
    Calculate shadow for a point given a light position.

    Computes the direction from point to light and calls soft shadow.
    Includes normal offset to prevent self-shadowing.

    Args:
        p: Surface point.
        n: Surface normal (for offset).
        light_pos: Light position.
        scene_sdf: Scene SDF function.
        config: Shadow configuration.
        normal_offset: Distance to offset along normal (default 0.01).

    Returns:
        Shadow factor [0.0, 1.0].
    """
    p_vec = Vec3Local(p[0], p[1], p[2])
    n_vec = Vec3Local(n[0], n[1], n[2]).normalized()
    light_vec = Vec3Local(light_pos[0], light_pos[1], light_pos[2])

    # Offset origin along normal to prevent self-shadowing
    ro = p_vec + n_vec * normal_offset

    # Direction to light
    rd = (light_vec - ro).normalized()

    return calculate_soft_shadow(ro.as_tuple(), rd.as_tuple(), scene_sdf, config)


# =============================================================================
# WGSL Code Generation
# =============================================================================

# Default WGSL function for soft shadow calculation
SHADOW_WGSL_FUNCTION = """\
/// Calculates soft shadow using Quilez's improved penumbra method.
///   ro         -- ray origin (surface point + offset)
///   rd         -- ray direction (towards light, normalized)
///   returns    -- shadow factor [0=shadow, 1=lit]
///
/// Reference: Inigo Quilez -- Soft shadows in raymarching
fn calculate_soft_shadow(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    let SHADOW_K: f32 = 16.0;
    let SHADOW_MAX_STEPS: i32 = 32;
    let SHADOW_MIN_DIST: f32 = 0.01;
    let SHADOW_MAX_DIST: f32 = 100.0;
    let SHADOW_MIN_STEP: f32 = 0.001;
    let SHADOW_MAX_STEP: f32 = 1.0;
    let SHADOW_EPSILON: f32 = 0.0001;

    var res: f32 = 1.0;
    var t: f32 = SHADOW_MIN_DIST;

    for (var i: i32 = 0; i < SHADOW_MAX_STEPS; i = i + 1) {
        let pos: vec3<f32> = ro + rd * t;
        let h: f32 = scene_sdf(pos).x;

        // Hard shadow: hit geometry
        if (h < SHADOW_EPSILON) {
            return 0.0;
        }

        // Soft shadow: contact hardening
        res = min(res, SHADOW_K * h / t);

        // Step forward
        t = t + clamp(h, SHADOW_MIN_STEP, SHADOW_MAX_STEP);

        // Beyond max distance
        if (t > SHADOW_MAX_DIST) {
            break;
        }
    }

    return clamp(res, 0.0, 1.0);
}"""


def generate_shadow_wgsl(config: Optional[ShadowConfig] = None) -> str:
    """
    Generate WGSL code for soft shadow calculation.

    Args:
        config: Shadow configuration to embed in shader. Uses defaults if None.

    Returns:
        WGSL function definition string.

    Example:
        >>> wgsl = generate_shadow_wgsl(ShadowConfig(k=32.0, max_steps=64))
        >>> assert "SHADOW_K: f32 = 32.0" in wgsl
        >>> assert "SHADOW_MAX_STEPS: i32 = 64" in wgsl
    """
    if config is None:
        config = ShadowConfig()

    return f"""\
/// Calculates soft shadow using Quilez's improved penumbra method.
///   ro         -- ray origin (surface point + offset)
///   rd         -- ray direction (towards light, normalized)
///   returns    -- shadow factor [0=shadow, 1=lit]
///
/// Reference: Inigo Quilez -- Soft shadows in raymarching
fn calculate_soft_shadow(ro: vec3<f32>, rd: vec3<f32>) -> f32 {{
    let SHADOW_K: f32 = {config.k};
    let SHADOW_MAX_STEPS: i32 = {config.max_steps};
    let SHADOW_MIN_DIST: f32 = {config.min_dist};
    let SHADOW_MAX_DIST: f32 = {config.max_dist};
    let SHADOW_MIN_STEP: f32 = {config.min_step};
    let SHADOW_MAX_STEP: f32 = {config.max_step};
    let SHADOW_EPSILON: f32 = {config.epsilon};

    var res: f32 = 1.0;
    var t: f32 = SHADOW_MIN_DIST;

    for (var i: i32 = 0; i < SHADOW_MAX_STEPS; i = i + 1) {{
        let pos: vec3<f32> = ro + rd * t;
        let h: f32 = scene_sdf(pos).x;

        // Hard shadow: hit geometry
        if (h < SHADOW_EPSILON) {{
            return 0.0;
        }}

        // Soft shadow: contact hardening
        res = min(res, SHADOW_K * h / t);

        // Step forward
        t = t + clamp(h, SHADOW_MIN_STEP, SHADOW_MAX_STEP);

        // Beyond max distance
        if (t > SHADOW_MAX_DIST) {{
            break;
        }}
    }}

    return clamp(res, 0.0, 1.0);
}}"""


def generate_shadow_wgsl_improved(config: Optional[ShadowConfig] = None) -> str:
    """
    Generate WGSL code for improved soft shadow (2020 version).

    Uses polynomial approximation for better penumbra estimation.

    Args:
        config: Shadow configuration.

    Returns:
        WGSL function definition string.
    """
    if config is None:
        config = ShadowConfig()

    return f"""\
/// Calculates soft shadow using Quilez's improved method (2020).
///   ro         -- ray origin
///   rd         -- ray direction
///   returns    -- shadow factor [0=shadow, 1=lit]
fn calculate_soft_shadow_improved(ro: vec3<f32>, rd: vec3<f32>) -> f32 {{
    let SHADOW_K: f32 = {config.k};
    let SHADOW_MAX_STEPS: i32 = {config.max_steps};
    let SHADOW_MIN_DIST: f32 = {config.min_dist};
    let SHADOW_MAX_DIST: f32 = {config.max_dist};
    let SHADOW_MIN_STEP: f32 = {config.min_step};
    let SHADOW_MAX_STEP: f32 = {config.max_step};
    let SHADOW_EPSILON: f32 = {config.epsilon};

    var res: f32 = 1.0;
    var t: f32 = SHADOW_MIN_DIST;
    var ph: f32 = 1e10;  // Previous height

    for (var i: i32 = 0; i < SHADOW_MAX_STEPS; i = i + 1) {{
        let pos: vec3<f32> = ro + rd * t;
        let h: f32 = scene_sdf(pos).x;

        // Hard shadow
        if (h < SHADOW_EPSILON) {{
            return 0.0;
        }}

        // Improved penumbra estimation
        let y: f32 = h * h / (2.0 * ph);
        let d: f32 = sqrt(max(0.0, h * h - y * y));
        res = min(res, SHADOW_K * d / max(0.0001, t - y));

        ph = h;
        t = t + clamp(h, SHADOW_MIN_STEP, SHADOW_MAX_STEP);

        if (t > SHADOW_MAX_DIST) {{
            break;
        }}
    }}

    return clamp(res, 0.0, 1.0);
}}"""


def generate_shadow_wgsl_inline(config: Optional[ShadowConfig] = None) -> str:
    """
    Generate inline WGSL code for shadows (for embedding in larger shaders).

    This version generates just the calculation logic without the function
    wrapper, suitable for inlining in compute shaders.

    Args:
        config: Shadow configuration parameters.

    Returns:
        WGSL code block (expects 'shadow_ro' and 'shadow_rd' in scope).
    """
    if config is None:
        config = ShadowConfig()

    return f"""\
// Soft shadow calculation (Quilez method)
var shadow_res: f32 = 1.0;
var shadow_t: f32 = {config.min_dist};
for (var shadow_i: i32 = 0; shadow_i < {config.max_steps}; shadow_i = shadow_i + 1) {{
    let shadow_pos: vec3<f32> = shadow_ro + shadow_rd * shadow_t;
    let shadow_h: f32 = scene_sdf(shadow_pos).x;
    if (shadow_h < {config.epsilon}) {{
        shadow_res = 0.0;
        break;
    }}
    shadow_res = min(shadow_res, {config.k} * shadow_h / shadow_t);
    shadow_t = shadow_t + clamp(shadow_h, {config.min_step}, {config.max_step});
    if (shadow_t > {config.max_dist}) {{
        break;
    }}
}}
let shadow: f32 = clamp(shadow_res, 0.0, 1.0);"""


# =============================================================================
# Scene Integration Helpers
# =============================================================================

def make_scene_shadow_evaluator(
    scene_sdf: Callable[[tuple[float, float, float]], float],
    config: Optional[ShadowConfig] = None,
) -> Callable[
    [tuple[float, float, float], tuple[float, float, float]], float
]:
    """
    Create a shadow evaluator function bound to a specific scene SDF.

    Args:
        scene_sdf: Scene SDF function.
        config: Shadow configuration.

    Returns:
        Function taking (ray_origin, ray_direction) and returning shadow factor.

    Example:
        >>> def sphere_sdf(p):
        ...     return math.sqrt(sum(x*x for x in p)) - 1.0
        >>> shadow_eval = make_scene_shadow_evaluator(sphere_sdf)
        >>> shadow = shadow_eval((2.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    """
    if config is None:
        config = ShadowConfig()

    def evaluator(
        ro: tuple[float, float, float],
        rd: tuple[float, float, float],
    ) -> float:
        return calculate_soft_shadow(ro, rd, scene_sdf, config)

    return evaluator


def make_light_shadow_evaluator(
    scene_sdf: Callable[[tuple[float, float, float]], float],
    light_pos: tuple[float, float, float],
    config: Optional[ShadowConfig] = None,
    normal_offset: float = 0.01,
) -> Callable[
    [tuple[float, float, float], tuple[float, float, float]], float
]:
    """
    Create a shadow evaluator for a specific light position.

    Args:
        scene_sdf: Scene SDF function.
        light_pos: Fixed light position.
        config: Shadow configuration.
        normal_offset: Distance to offset along normal.

    Returns:
        Function taking (point, normal) and returning shadow factor.
    """
    if config is None:
        config = ShadowConfig()

    def evaluator(
        p: tuple[float, float, float],
        n: tuple[float, float, float],
    ) -> float:
        return calculate_shadow_from_light(
            p, n, light_pos, scene_sdf, config, normal_offset
        )

    return evaluator


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    "ShadowConfig",
    # Core functions
    "calculate_soft_shadow",
    "calculate_soft_shadow_improved",
    "calculate_hard_shadow",
    "calculate_shadow_from_light",
    # WGSL generation
    "SHADOW_WGSL_FUNCTION",
    "generate_shadow_wgsl",
    "generate_shadow_wgsl_improved",
    "generate_shadow_wgsl_inline",
    # Helpers
    "make_scene_shadow_evaluator",
    "make_light_shadow_evaluator",
    "Vec3Local",
]
